"""
hotpatch_caps.py

HotPatchCaps: A Capsule Network with Runtime Hot Patching
for Zero-Day API Attack Detection.

Reference:
    "HotPatchCaps: A Capsule Network with Runtime Hot Patching
     for Zero-Day API Attack Detections", HotPatchCaps_TNSM.

Architecture overview (paper Section V-D):
    x_stat : TF-IDF (1-3-gram, max_features=1500, L2-normalized)
             -> MLP (512->256, ReLU, LayerNorm, Dropout=0.1)
             -> 32 primary capsules (8-dim each)
             -> SlotControlledCapsuleLayer -> K class capsules (16-dim)

    x_cue  : 50-dim expert semantic cues (see features/expert_features.py)
             z-scored on training set, right-padded to 100-slot layout
             -> used ONLY as slot routing priors (NOT concatenated with x_stat)

    logits : temperature-scaled cosine(class_caps, target_vectors)

Slot-controlled routing (Eq. 7 / Listing 1):
    b_ij <- b_ij + sigma_k  alpha_k * M_k(r) * w_{k->j}
    applied before softmax; prior strength decays 0.02 per routing iteration.

Hot-patch API  (Section IV-D):
    add_slot / update_slot / enable / disable / rollback / dry_run
    via HotPatchManager — no model retraining required.

Unknown detection (MSP gating, Eq. 6):
    u(x) = 1 - max_k cos(class_cap_k, target_k)
    threshold tau = (1 - FPR) quantile on a held-out validation set.
"""

import ast
import copy
import json
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Hyper-parameters matching paper Section V-D
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 1500
TFIDF_NGRAM_RANGE  = (1, 3)
NUM_SLOTS          = 100      # 50 active cue slots + 50 reserved for patches
NUM_CUE_FEATURES   = 50       # expert semantic cues (Table III)
NUM_PRIMARY_CAPS   = 32
PRIMARY_CAP_DIM    = 8
CLASS_CAP_DIM      = 16
ROUTING_ITERS      = 3
PRIOR_STRENGTH     = 0.1      # initial slot prior injection strength
PRIOR_DECAY        = 0.02     # strength decay per routing iteration
TEMPERATURE_INIT   = 30.0     # learnable logit temperature
DROPOUT_P          = 0.1
ENTROPY_REG_W      = 1e-3
MARGIN             = 0.4
TARGET_SIM         = 0.8
LABEL_SMOOTH_EPS   = 0.05
UNKNOWN_FPR        = 0.01
SEED               = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Section 1: Capsule Layer Components
# ===========================================================================

class PrimaryCapsLayer(nn.Module):
    """
    32 primary capsules (8-dim each) projected from a 256-dim MLP output.
    Output: [B, num_capsules, capsule_dim]
    """

    def __init__(self, in_channels: int, num_capsules: int, capsule_dim: int):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, capsule_dim),
                nn.LayerNorm(capsule_dim),
            )
            for _ in range(num_capsules)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([cap(x) for cap in self.capsules], dim=1)


class SlotControlledCapsuleLayer(nn.Module):
    """
    Dynamic-routing capsule layer augmented with slot-controlled priors.

    Training-time priors  : z-scored 50-dim cue features padded to 100-dim,
                            multiplied by learnable slot_routing_priors [100, K].
    Inference-time patches: (M_k, alpha_k, w_{k->j}) triples, added to routing
                            logits b_ij before coupling-coefficient softmax.

    Prior strength decays by PRIOR_DECAY each routing iteration:
        iter 0 : PRIOR_STRENGTH       (0.10)
        iter 1 : PRIOR_STRENGTH - 0.02 (0.08)
        iter 2 : PRIOR_STRENGTH - 0.04 (0.06)
    """

    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,
        in_channels: int,
        out_channels: int,
        num_slots: int = NUM_SLOTS,
    ):
        super().__init__()
        self.num_capsules    = num_capsules
        self.num_route_nodes = num_route_nodes
        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.num_slots       = num_slots

        self.weights = nn.Parameter(
            torch.randn(num_route_nodes, num_capsules, out_channels, in_channels) * 0.1
        )
        # Slot routing priors: [num_slots, K], init=0 (neutral)
        self.slot_routing_priors = nn.Parameter(
            torch.zeros(num_slots, num_capsules)
        )
        self.last_coupling: Optional[torch.Tensor] = None

    @staticmethod
    def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale   = sq_norm / (1.0 + sq_norm)
        return scale * tensor / torch.sqrt(sq_norm + 1e-8)

    def forward(
        self,
        x: torch.Tensor,
        slot_features: Optional[torch.Tensor] = None,
        hot_patches: Optional[List] = None,
        request_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            x             : [B, num_route_nodes, in_channels]
            slot_features : [B, num_slots]  z-scored cue vector (training prior)
            hot_patches   : list of PatchSlot, applied at inference
            request_text  : raw request string for matcher evaluation

        Returns:
            [B, num_capsules, out_channels]
        """
        B      = x.size(0)
        device = x.device
        dtype  = x.dtype

        predictions = torch.einsum('bni,njoi->bnjo', x, self.weights)
        b = torch.zeros(B, self.num_route_nodes, self.num_capsules, 1,
                        device=device, dtype=dtype)

        # --- Slot-controlled prior from cue features (Section V-D) ----------
        slot_prior = None
        if slot_features is not None:
            sf = slot_features.to(device=device, dtype=dtype)
            if sf.shape[1] < self.num_slots:
                pad = torch.zeros(B, self.num_slots - sf.shape[1],
                                  device=device, dtype=dtype)
                sf = torch.cat([sf, pad], dim=1)
            elif sf.shape[1] > self.num_slots:
                sf = sf[:, :self.num_slots]

            sf_relu = torch.relu(sf)
            sf_w    = sf_relu / sf_relu.sum(dim=1, keepdim=True).clamp_min(1e-6)
            sf_w    = sf_w - (1.0 / self.num_slots)   # center

            slot_prior = torch.matmul(sf_w, self.slot_routing_priors)  # [B, K]
            slot_prior = slot_prior.unsqueeze(1).unsqueeze(-1)          # [B,1,K,1]
            slot_prior = slot_prior.expand(-1, self.num_route_nodes, -1, -1)
            b = b + slot_prior * PRIOR_STRENGTH

        # --- Hot-patch priors (Listing 1, Eq. 7) ----------------------------
        if hot_patches and request_text is not None:
            for patch in hot_patches:
                if not patch.enabled:
                    continue
                m = patch.matcher_fn(request_text)
                if m > 0:
                    w = torch.tensor(patch.w_to_class, dtype=dtype, device=device)
                    delta = (patch.alpha * m) * w.view(1, 1, self.num_capsules, 1)
                    b = b + delta

        # --- Dynamic routing with decaying prior re-injection ----------------
        c = None
        for i in range(ROUTING_ITERS):
            c = F.softmax(b, dim=2)
            s = (predictions * c).sum(dim=1, keepdim=True)
            v = self.squash(s, dim=-1)

            if i < ROUTING_ITERS - 1:
                b = b + (predictions * v).sum(dim=-1, keepdim=True)
                if slot_prior is not None:
                    strength_i = max(0.0, PRIOR_STRENGTH - PRIOR_DECAY * (i + 1))
                    b = b + slot_prior * strength_i

        self.last_coupling = c
        return v.squeeze(1)   # [B, K, out_channels]

    def coupling_entropy(self) -> torch.Tensor:
        """Low-entropy routing regularizer (encourages sparse routing)."""
        if self.last_coupling is None:
            return torch.tensor(0.0)
        c = self.last_coupling.squeeze(-1).clamp_min(1e-12)
        return -(c * c.log()).sum(dim=2).mean()


# ===========================================================================
# Section 2: Loss and Full Model
# ===========================================================================

class CosineSimilarityLoss(nn.Module):
    """
    Margin-based cosine loss with label smoothing (paper Section V-C).

    Target vectors are fixed unit vectors (one per class).
    Positive term : cos(cap_k, target_k) >= target_sim (0.8)
    Negative term : cos(cap_j, target_j) < target_sim - margin (0.4)
    """

    def __init__(self, num_labels: int, cap_dim: int,
                 margin: float = MARGIN, target_sim: float = TARGET_SIM):
        super().__init__()
        self.margin     = margin
        self.target_sim = target_sim
        targets = F.normalize(torch.randn(num_labels, cap_dim), dim=1)
        self.register_buffer('targets', targets)

    def forward(self, caps: torch.Tensor, labels: torch.Tensor,
                eps: float = LABEL_SMOOTH_EPS) -> torch.Tensor:
        B, K, _ = caps.shape
        caps_norm = F.normalize(caps, dim=2)
        cos_sim   = (caps_norm * self.targets.unsqueeze(0)).sum(dim=2)  # [B,K]

        pos_mask = F.one_hot(labels, K).float()
        pos_loss = (self.target_sim - cos_sim).clamp_min(0) * pos_mask
        neg_loss = (cos_sim - (self.target_sim - self.margin)).clamp_min(0) * (1 - pos_mask)

        smooth = pos_mask * (1 - eps) + eps / K
        ce_loss = -(smooth * F.log_softmax(cos_sim, dim=1)).sum(dim=1).mean()
        return (pos_loss + neg_loss).sum(dim=1).mean() + ce_loss

    def cosine_logits(self, caps: torch.Tensor) -> torch.Tensor:
        return (F.normalize(caps, dim=2) * self.targets.unsqueeze(0)).sum(dim=2)


class HotPatchCapsModel(nn.Module):
    """
    HotPatchCaps full model (Section IV and V-D).

    Input:
        features      : [B, TFIDF_MAX_FEATURES]   L2-normalised TF-IDF
        slot_features : [B, NUM_SLOTS]             z-scored cue vector

    The cue vector is injected ONLY into routing logits, never concatenated
    with the lexical feature stream.
    """

    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.input_dim  = input_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(DROPOUT_P),
        )
        self.primary_caps = PrimaryCapsLayer(256, NUM_PRIMARY_CAPS, PRIMARY_CAP_DIM)
        self.digit_caps   = SlotControlledCapsuleLayer(
            num_capsules=num_labels,
            num_route_nodes=NUM_PRIMARY_CAPS,
            in_channels=PRIMARY_CAP_DIM,
            out_channels=CLASS_CAP_DIM,
        )
        self.cos_loss   = CosineSimilarityLoss(num_labels, CLASS_CAP_DIM)
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE_INIT))
        self.caps_drop  = nn.Dropout(DROPOUT_P)

    def forward(
        self,
        features: torch.Tensor,
        slot_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        hot_patches: Optional[List] = None,
        request_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Returns (loss, logits).  loss is None at inference.
        logits: [B, K]  (cosine similarities * temperature)
        """
        x    = self.embedding(features)
        prim = self.primary_caps(x)

        if hot_patches and request_texts is not None:
            caps_list = []
            for i, rtxt in enumerate(request_texts):
                cap = self.digit_caps(
                    prim[i:i+1],
                    slot_features[i:i+1] if slot_features is not None else None,
                    hot_patches=hot_patches,
                    request_text=rtxt,
                )
                caps_list.append(cap)
            class_caps = torch.cat(caps_list, dim=0)
        else:
            class_caps = self.digit_caps(prim, slot_features)

        class_caps = self.caps_drop(class_caps)
        logits     = self.cos_loss.cosine_logits(class_caps) * self.temperature

        loss = None
        if labels is not None:
            cos_loss = self.cos_loss(class_caps, labels)
            ent_reg  = self.digit_caps.coupling_entropy() * ENTROPY_REG_W
            l2_reg   = sum(p.pow(2).sum() for p in self.parameters()) * 1e-5
            loss     = cos_loss + ent_reg + l2_reg

        return loss, logits

    def unknown_score(self, class_caps: torch.Tensor) -> torch.Tensor:
        """MSP uncertainty: u(x) = 1 - max_k cos(cap_k, target_k)."""
        return 1.0 - self.cos_loss.cosine_logits(class_caps).max(dim=1).values


# ===========================================================================
# Section 3: Hot-Patch Management (Section IV-D)
# ===========================================================================

@dataclass
class PatchSlot:
    """
    A single runtime hot-patch slot.

    Attributes:
        name       : human-readable identifier
        matcher_fn : callable(request_text) -> float  (0 = no match)
        alpha      : prior strength
        w_to_class : weight vector of length K (one per class capsule)
        scope      : optional scope string (endpoint, tenant, time window)
        enabled    : toggle without deletion
        version    : increments on each update (for rollback)
    """
    name:       str
    matcher_fn: Callable[[str], float]
    alpha:      float
    w_to_class: List[float]
    scope:      Optional[str]  = None
    enabled:    bool           = True
    version:    int            = 1
    _history:   List[dict]     = field(default_factory=list, repr=False)


class HotPatchManager:
    """
    Operator-facing API for managing runtime hot-patch slots.

    All operations are zero-downtime; the model does not need to be retrained.

    Example usage::

        mgr = HotPatchManager(num_labels=7)

        # Register a Log4Shell detector
        slot_id = mgr.add_slot(
            name='Log4Shell',
            matcher_fn=lambda text: 1.0 if '${jndi:' in text.lower() else 0.0,
            alpha=0.5,
            w_to_class=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # points to Log4J class
        )

        # Test without committing
        mgr.dry_run(slot_id, sample_request_text)

        # Disable temporarily
        mgr.disable(slot_id)

        # Rollback to previous version after an update
        mgr.update_slot(slot_id, alpha=0.3)
        mgr.rollback(slot_id)
    """

    def __init__(self, num_labels: int):
        self.num_labels  = num_labels
        self._slots:     Dict[str, PatchSlot] = {}
        self._id_counter = 0

    def add_slot(
        self,
        name: str,
        matcher_fn: Callable[[str], float],
        alpha: float,
        w_to_class: Optional[List[float]] = None,
        scope: Optional[str] = None,
    ) -> str:
        """Add a new slot; returns its slot_id."""
        if w_to_class is None:
            # Uniform weight across all attack classes (exclude Benign at 0)
            w_to_class = [0.0] + [1.0 / (self.num_labels - 1)] * (self.num_labels - 1)
        assert len(w_to_class) == self.num_labels
        slot_id = f"patch_{self._id_counter:04d}"
        self._id_counter += 1
        self._slots[slot_id] = PatchSlot(
            name=name, matcher_fn=matcher_fn,
            alpha=alpha, w_to_class=w_to_class, scope=scope,
        )
        print(f"[HotPatch] + Added '{name}' ({slot_id})  alpha={alpha}  scope={scope}")
        return slot_id

    def update_slot(self, slot_id: str, **fields) -> None:
        """Update matcher_fn, alpha, or w_to_class; old values are saved for rollback."""
        slot = self._get(slot_id)
        slot._history.append({
            'version':    slot.version,
            'alpha':      slot.alpha,
            'w_to_class': copy.deepcopy(slot.w_to_class),
        })
        for k, v in fields.items():
            setattr(slot, k, v)
        slot.version += 1
        print(f"[HotPatch] Updated {slot_id} -> version {slot.version}")

    def enable(self, slot_id: str)  -> None:
        self._get(slot_id).enabled = True
        print(f"[HotPatch] Enabled  {slot_id}")

    def disable(self, slot_id: str) -> None:
        self._get(slot_id).enabled = False
        print(f"[HotPatch] Disabled {slot_id}")

    def rollback(self, slot_id: str) -> None:
        """Atomic rollback to the most recent saved version."""
        slot = self._get(slot_id)
        if not slot._history:
            print(f"[HotPatch] No history for {slot_id}")
            return
        prev = slot._history.pop()
        slot.alpha      = prev['alpha']
        slot.w_to_class = prev['w_to_class']
        slot.version    = prev['version']
        print(f"[HotPatch] Rolled back {slot_id} to version {slot.version}")

    def dry_run(self, slot_id: str, sample_text: str) -> Dict:
        """Evaluate slot on a sample without committing to the model."""
        slot  = self._get(slot_id)
        score = slot.matcher_fn(sample_text)
        return {
            'slot_id':         slot_id,
            'name':            slot.name,
            'match_score':     score,
            'would_fire':      score > 0,
            'effective_alpha': slot.alpha * score if score > 0 else 0.0,
        }

    def active_slots(self) -> List[PatchSlot]:
        return [s for s in self._slots.values() if s.enabled]

    def summary(self) -> None:
        n_on = sum(s.enabled for s in self._slots.values())
        print(f"\n[HotPatch] {n_on}/{len(self._slots)} slots active")
        for sid, s in self._slots.items():
            state = "ON " if s.enabled else "OFF"
            print(f"  [{state}] {sid}  '{s.name}'  alpha={s.alpha:.3f}  "
                  f"v{s.version}  scope={s.scope}")

    def _get(self, slot_id: str) -> PatchSlot:
        if slot_id not in self._slots:
            raise KeyError(f"Slot '{slot_id}' not found")
        return self._slots[slot_id]


# ===========================================================================
# Section 4: Dataset and Preprocessing
# ===========================================================================

def _parse_headers(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {'raw': raw}
    return {}


def preprocess_example(example: dict, label_map: dict) -> dict:
    """Convert a raw dataset dict to the feature dict used by HotPatchCapsDataset."""
    from src.features.expert_features import extract_cue_features
    req     = example.get('request', example)
    method  = req.get('method', '')
    url     = req.get('url', '')
    headers = _parse_headers(req.get('headers', {}))
    body    = req.get('body', '')
    text    = f"{method} {url} Headers: {headers} Body: {body}"

    attack_type = req.get('Attack_Tag', None)
    label = (label_map.get(attack_type, label_map.get('Benign', 0))
             if attack_type else label_map.get('Benign', 0))

    return {
        'text':        text,
        'labels':      label,
        'cue_features': extract_cue_features(method, url, headers, body),
        'attack_type': attack_type or 'Benign',
    }


class HotPatchCapsDataset(Dataset):
    """PyTorch Dataset: TF-IDF + z-scored slot features."""

    def __init__(self, examples: list, vectorizer: TfidfVectorizer,
                 cue_scaler: StandardScaler):
        self.examples = examples
        texts = [e['text'] for e in examples]

        X     = vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
        self.tfidf = X / norms

        cues = np.array([e['cue_features'] for e in examples], dtype=np.float32)
        cues = cue_scaler.transform(cues)
        pad  = np.zeros((len(cues), NUM_SLOTS - NUM_CUE_FEATURES), dtype=np.float32)
        self.slots = np.concatenate([cues, pad], axis=1)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        e = self.examples[idx]
        return {
            'features':     torch.tensor(self.tfidf[idx],  dtype=torch.float32),
            'slot_features': torch.tensor(self.slots[idx],  dtype=torch.float32),
            'labels':       torch.tensor(e['labels'],       dtype=torch.long),
            'text':         e['text'],
            'attack_type':  e['attack_type'],
        }


def collate_fn(batch: list) -> dict:
    return {
        'features':     torch.stack([b['features']     for b in batch]),
        'slot_features': torch.stack([b['slot_features'] for b in batch]),
        'labels':       torch.stack([b['labels']       for b in batch]),
        'texts':        [b['text']        for b in batch],
        'attack_types': [b['attack_type'] for b in batch],
    }


# ===========================================================================
# Section 5: Training and Evaluation
# ===========================================================================

def calibrate_unknown_threshold(
    model: HotPatchCapsModel,
    val_loader: DataLoader,
    device: torch.device,
    target_fpr: float = UNKNOWN_FPR,
) -> float:
    """
    Set tau so that the false-positive rate on known-class validation data
    equals target_fpr (paper: default 1%).
    """
    model.eval()
    u_scores = []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch['features'].to(device)
            slots = batch['slot_features'].to(device)
            x     = model.embedding(feats)
            prim  = model.primary_caps(x)
            caps  = model.digit_caps(prim, slots)
            u     = model.unknown_score(caps)
            u_scores.append(u.cpu().numpy())
    u_all = np.concatenate(u_scores)
    tau   = float(np.quantile(u_all, 1.0 - target_fpr))
    print(f"[Calibration] tau={tau:.4f}  FPR_target={target_fpr:.2%}  n={len(u_all)}")
    return tau


def train_one_epoch(
    model: HotPatchCapsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        feats  = batch['features'].to(device)
        slots  = batch['slot_features'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        loss, _ = model(feats, slots, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: HotPatchCapsModel,
    loader: DataLoader,
    device: torch.device,
    id_to_label: dict,
    tau: Optional[float] = None,
    hot_patch_mgr: Optional[HotPatchManager] = None,
) -> dict:
    """
    Full evaluation pass.

    Returns a dict with accuracy, macro P/R/F1, confusion matrix, and
    Unknown-detection metrics (if tau is set).
    """
    model.eval()
    all_preds, all_labels = [], []
    patches = hot_patch_mgr.active_slots() if hot_patch_mgr else None

    with torch.no_grad():
        for batch in loader:
            feats  = batch['features'].to(device)
            slots  = batch['slot_features'].to(device)
            labels = batch['labels'].to(device)
            texts  = batch['texts']

            _, logits = model(feats, slots,
                              hot_patches=patches,
                              request_texts=texts if patches else None)
            preds = logits.argmax(dim=1)

            if tau is not None:
                x    = model.embedding(feats)
                prim = model.primary_caps(x)
                caps = model.digit_caps(prim, slots)
                u    = model.unknown_score(caps)
                preds[u >= tau] = -1

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    known_mask = y_pred >= 0
    n_unknown  = int((~known_mask).sum())

    acc = (y_pred[known_mask] == y_true[known_mask]).mean() if known_mask.any() else 0.0
    prec, rec, f1, _ = (
        precision_recall_fscore_support(
            y_true[known_mask], y_pred[known_mask],
            average='macro', zero_division=0
        ) if known_mask.any() else (0.0, 0.0, 0.0, None)
    )

    num_labels = model.num_labels
    extra      = [-1] if n_unknown > 0 else []
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)) + extra)

    return {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'n_unknown': n_unknown,
        'confusion': cm,
        'y_pred':    y_pred,
        'y_true':    y_true,
    }


# ===========================================================================
# Section 6: Single-Request Inference
# ===========================================================================

def predict_with_hotpatch(
    model: HotPatchCapsModel,
    vectorizer: TfidfVectorizer,
    cue_scaler: StandardScaler,
    request_data: dict,
    id_to_label: dict,
    hot_patch_mgr: Optional[HotPatchManager] = None,
    tau: Optional[float] = None,
) -> dict:
    """
    Single-request inference following paper Listing 1 / Algorithm 1:

    1. Tokenize -> x_stat (TF-IDF), x_cue (50-dim cue vector)
    2. Forward through capsule encoder with base slot priors
    3. Inject active hot-patch priors into routing logits
    4. MSP gating: classify as Unknown if u(x) >= tau

    Args:
        request_data : dict with keys 'method', 'url', 'headers', 'body'
        id_to_label  : mapping from integer label to class name string
        hot_patch_mgr: HotPatchManager with active patch slots
        tau          : Unknown detection threshold (from calibrate_unknown_threshold)

    Returns:
        dict with prediction, confidence, unknown_score, and which patches fired
    """
    from src.features.expert_features import extract_cue_features
    device = next(model.parameters()).device
    model.eval()

    method  = request_data.get('method', 'GET')
    url     = request_data.get('url', '')
    headers = request_data.get('headers', {})
    body    = request_data.get('body', '')
    text    = f"{method} {url} Headers: {headers} Body: {body}"

    arr  = vectorizer.transform([text]).toarray().astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    tfidf_t = torch.tensor(arr, dtype=torch.float32).to(device)

    cues    = extract_cue_features(method, url, headers, body)
    cue_arr = cue_scaler.transform(np.array([cues], dtype=np.float32))
    pad     = np.zeros((1, NUM_SLOTS - NUM_CUE_FEATURES), dtype=np.float32)
    slot_t  = torch.tensor(np.concatenate([cue_arr, pad], axis=1),
                           dtype=torch.float32).to(device)

    patches = hot_patch_mgr.active_slots() if hot_patch_mgr else None

    with torch.no_grad():
        _, logits = model(tfidf_t, slot_t,
                          hot_patches=patches,
                          request_texts=[text] if patches else None)
        x    = model.embedding(tfidf_t)
        prim = model.primary_caps(x)
        caps = model.digit_caps(prim, slot_t,
                                hot_patches=patches,
                                request_text=text if patches else None)
        u = model.unknown_score(caps).item()

    pred_id    = logits.argmax(dim=1).item()
    probs      = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_label = id_to_label.get(pred_id, str(pred_id))
    if tau is not None and u >= tau:
        pred_label = 'Unknown'

    fired = [p.name for p in (patches or []) if p.matcher_fn(text) > 0]

    return {
        'prediction':    pred_label,
        'confidence':    float(probs[pred_id]),
        'unknown_score': u,
        'is_unknown':    (tau is not None) and (u >= tau),
        'class_probs':   {id_to_label.get(i, str(i)): float(p)
                          for i, p in enumerate(probs)},
        'fired_patches': fired,
    }


# ===========================================================================
# Section 7: Dataset Utilities
# ===========================================================================

def stratified_split(
    examples: list,
    test_size: float = 0.20,
    seed: int = SEED,
) -> Tuple[list, list]:
    """Split into train/test while preserving per-class ratios."""
    rng     = random.Random(seed)
    buckets: Dict[int, list] = defaultdict(list)
    for ex in examples:
        buckets[ex['labels']].append(ex)

    train_out, test_out = [], []
    for exs in buckets.values():
        rng.shuffle(exs)
        n_test = max(1, int(len(exs) * test_size))
        test_out.extend(exs[:n_test])
        train_out.extend(exs[n_test:])

    rng.shuffle(train_out)
    rng.shuffle(test_out)
    return train_out, test_out


def balanced_sample(
    examples: list,
    label_map: dict,
    strategy: str = 'undersample',
) -> list:
    """
    Reduce class imbalance by downsampling the Benign class.

    strategy:
        'undersample' : target = average attack class count
        'equal'       : target = minimum class count
    """
    rng     = random.Random(SEED)
    buckets: Dict[int, list] = defaultdict(list)
    for ex in examples:
        buckets[ex['labels']].append(ex)

    benign_id = label_map.get('Benign', 0)
    attack_counts = [len(v) for k, v in buckets.items() if k != benign_id]
    if not attack_counts:
        return examples

    target = min(attack_counts) if strategy == 'equal' else int(np.mean(attack_counts))

    out = []
    for label_id, exs in buckets.items():
        if label_id == benign_id:
            out.extend(rng.sample(exs, min(target, len(exs))))
        else:
            out.extend(exs)
    rng.shuffle(out)
    return out
