"""
expert_features.py

50-dimensional expert semantic cue vector (Table III of the paper).

This module implements domain-agnostic, hand-crafted features used
as routing priors in HotPatchCaps. The cues capture structural and
syntactic patterns associated with common web attack classes:

    SQLi        (indices  0-9,  10 features)
    XSS         (indices 10-20, 11 features)
    DirTrav     (indices 21-25,  5 features)
    LOG4J/JNDI  (indices 26-33,  8 features)
    LogForging  (indices 34-38,  5 features)
    JNDI placmt (indices 39-44,  6 features)
    Browser/Ctx (indices 45-48,  4 features)
    Request shp (index   49,     1 feature )

These features are injected only into the capsule routing layer, never
concatenated with the TF-IDF feature stream, which preserves the
statistical independence of the two input modalities.

Usage::

    feats = extract_cue_features(method, url, headers, body)
    # feats is a list of 50 floats
"""

import re
from typing import Dict, List, Union

NUM_CUE_FEATURES = 50   # must match paper Table III


def extract_cue_features(
    method: str,
    url: str,
    headers: Union[dict, str],
    body: str,
) -> List[float]:
    """
    Extract the 50-dimensional expert semantic cue vector.

    Args:
        method  : HTTP method string (e.g. 'GET', 'POST')
        url     : request URL (full or path only)
        headers : request headers as dict or raw string
        body    : request body as string

    Returns:
        List of 50 floats, one per cue dimension.
    """
    if isinstance(headers, dict):
        hdr_str = " ".join(f"{k}: {v}" for k, v in headers.items())
    elif isinstance(headers, str):
        hdr_str = headers
    else:
        hdr_str = str(headers) if headers else ""

    full   = f"{method} {url} {hdr_str} {body}"
    full_l = full.lower()
    url_l  = url.lower()
    body_l = body.lower()
    hdr_l  = hdr_str.lower()

    feats: List[float] = []

    # ------------------------------------------------------------------
    # SQLi  (0-9, 10 features)
    # ------------------------------------------------------------------
    sql_kw = ['select', 'union', 'insert', 'update', 'delete', 'drop', 'alter',
              'execute', 'exec', 'declare', 'cast', 'waitfor', 'delay']
    feats.append(float(min(sum(1 for k in sql_kw if k in full_l), 5)))   # 0  keyword count
    feats.append(1.0 if re.search(r'\bor\b\s*\w*\s*=', full_l) else 0.0) # 1  OR tautology
    feats.append(1.0 if re.search(r'\band\b\s*\w*\s*=', full_l) else 0.0)# 2  AND tautology
    feats.append(1.0 if '--' in full else 0.0)                            # 3  line comment
    feats.append(1.0 if '#' in full else 0.0)                             # 4  hash comment
    feats.append(1.0 if ';' in full else 0.0)                             # 5  statement sep
    sq = full.count("'")
    dq = full.count('"')
    feats.append(float(min(sq, 10)))                                       # 6  single-quote count
    feats.append(float(min(dq, 10)))                                       # 7  double-quote count
    feats.append(1.0 if (sq % 2 == 1 or dq % 2 == 1) else 0.0)           # 8  unbalanced quotes
    feats.append(1.0 if re.search(r'sleep\s*\(', full_l) else 0.0)        # 9  time-based blind

    # ------------------------------------------------------------------
    # XSS  (10-20, 11 features)
    # ------------------------------------------------------------------
    feats.append(1.0 if re.search(r'<script', full_l) else 0.0)           # 10
    feats.append(1.0 if '</script>' in full_l else 0.0)                   # 11
    feats.append(1.0 if 'javascript:' in full_l else 0.0)                 # 12
    feats.append(1.0 if 'eval(' in full_l else 0.0)                       # 13
    feats.append(1.0 if 'document.' in full_l else 0.0)                   # 14
    ev_handlers = ['onload', 'onclick', 'onmouseover', 'onfocus', 'onerror']
    feats.append(float(min(sum(1 for e in ev_handlers if e in full_l), 5)))# 15 event handler count
    feats.append(1.0 if '&#' in full else 0.0)                            # 16 HTML entity
    feats.append(1.0 if '%3c' in full_l else 0.0)                         # 17 URL-encoded <
    feats.append(1.0 if '%3e' in full_l else 0.0)                         # 18 URL-encoded >
    feats.append(1.0 if 'alert(' in full_l else 0.0)                      # 19
    dom_sinks = ['innerhtml', 'outerhtml', 'document.write']
    feats.append(1.0 if any(s in full_l for s in dom_sinks) else 0.0)     # 20 DOM sink

    # ------------------------------------------------------------------
    # Directory Traversal  (21-25, 5 features)
    # ------------------------------------------------------------------
    feats.append(1.0 if '../' in full else 0.0)                           # 21
    feats.append(1.0 if '..\\ ' in full else 0.0)                         # 22
    feats.append(1.0 if '%2e%2e%2f' in full_l else 0.0)                  # 23 URL-encoded ../
    sens_paths = ['/etc/', '/var/', '/bin/', '/usr/', 'passwd', 'shadow', 'config']
    feats.append(float(min(sum(1 for p in sens_paths if p in full_l), 3)))# 24 sensitive path
    feats.append(float(min(full.count('../'), 5)))                         # 25 traversal depth

    # ------------------------------------------------------------------
    # Log4Shell / JNDI Injection  (26-33, 8 features)
    # ------------------------------------------------------------------
    feats.append(1.0 if re.search(r'\$\{jndi:', full, re.I) else 0.0)     # 26 JNDI lookup
    feats.append(1.0 if re.search(r'(ldap|rmi|dns)://', full, re.I) else 0.0) # 27 remote scheme
    feats.append(1.0 if re.search(r'\$\{[^}]*\$\{', full) else 0.0)       # 28 nested interp.
    feats.append(1.0 if re.search(r'\$\{env:', full, re.I) else 0.0)       # 29 env var lookup
    feats.append(1.0 if re.search(r'\$\{sys:', full, re.I) else 0.0)       # 30 sys prop lookup
    feats.append(1.0 if re.search(r'\$\{jndi:', hdr_str, re.I) else 0.0)  # 31 JNDI in header
    rare_proto = ['corba', 'iiop', 'nds', 'coap', 'nis']
    feats.append(1.0 if any(p + '://' in full_l for p in rare_proto) else 0.0) # 32 rare proto
    obf = re.search(r'\$\{[^}]*(lower|upper|:-)[^}]*jndi', full, re.I)
    feats.append(1.0 if obf else 0.0)                                      # 33 obfuscated JNDI

    # ------------------------------------------------------------------
    # Log Forging  (34-38, 5 features)
    # ------------------------------------------------------------------
    feats.append(1.0 if re.search(r'\n|\r', full) else 0.0)               # 34 CRLF injection
    feats.append(1.0 if ('%0a' in full_l or '%0d' in full_l) else 0.0)    # 35 URL-encoded CRLF
    env_vars = ['$user', '$home', '$path', '$shell']
    feats.append(float(min(sum(1 for v in env_vars if v in full_l), 3)))  # 36 env var ref
    feats.append(1.0 if ('cookie' in hdr_l and
                          re.search(r'\$\{jndi:', full, re.I)) else 0.0)  # 37 cookie + JNDI
    log_tokens = ['error', 'warn', 'info', 'debug', 'fatal', 'trace']
    feats.append(1.0 if any(t in full_l for t in log_tokens) else 0.0)    # 38 log level token

    # ------------------------------------------------------------------
    # JNDI Placement  (39-44, 6 features)
    # ------------------------------------------------------------------
    jndi_url  = bool(re.search(r'\$\{jndi:', url, re.I))
    jndi_body = bool(re.search(r'\$\{jndi:', body, re.I))
    jndi_hdr  = bool(re.search(r'\$\{jndi:', hdr_str, re.I))
    feats.append(1.0 if jndi_url  else 0.0)                               # 39 JNDI in URL
    feats.append(1.0 if jndi_body else 0.0)                               # 40 JNDI in body
    feats.append(1.0 if jndi_hdr  else 0.0)                               # 41 JNDI in header
    sec_fetch = 'sec-fetch' in hdr_l
    feats.append(1.0 if (sec_fetch and (jndi_url or jndi_body or jndi_hdr)) else 0.0) # 42
    jndi_count = sum([jndi_url, jndi_body, jndi_hdr])
    feats.append(1.0 if (jndi_hdr and jndi_count == 1) else 0.0)          # 43 header-only JNDI
    feats.append(1.0 if jndi_count > 1 else 0.0)                          # 44 multi-location

    # ------------------------------------------------------------------
    # Browser / Context  (45-48, 4 features)
    # ------------------------------------------------------------------
    feats.append(1.0 if 'mozilla/' in hdr_l else 0.0)                     # 45 browser UA
    feats.append(1.0 if 'sec-fetch-dest: document' in hdr_l else 0.0)     # 46 fetch-dest
    feats.append(1.0 if 'text/html' in hdr_l else 0.0)                    # 47 HTML accept
    forum_tokens = ['post', 'comment', 'thread', 'reply', 'forum']
    feats.append(1.0 if any(t in full_l for t in forum_tokens) else 0.0)  # 48 forum context

    # ------------------------------------------------------------------
    # Request Shape  (49, 1 feature)
    # ------------------------------------------------------------------
    feats.append(1.0 if '?' in url else 0.0)                              # 49 has query string

    assert len(feats) == NUM_CUE_FEATURES, \
        f"Feature count mismatch: expected {NUM_CUE_FEATURES}, got {len(feats)}"
    return feats


# ---------------------------------------------------------------------------
# Feature group metadata (useful for ablation studies and interpretability)
# ---------------------------------------------------------------------------
CUE_GROUPS: Dict[str, List[int]] = {
    'SQLi':         list(range(0, 10)),
    'XSS':          list(range(10, 21)),
    'DirTrav':      list(range(21, 26)),
    'Log4Shell':    list(range(26, 34)),
    'LogForging':   list(range(34, 39)),
    'JNDI_Placmt':  list(range(39, 45)),
    'Browser_Ctx':  list(range(45, 49)),
    'Request_Shape': [49],
}

CUE_NAMES: List[str] = [
    # SQLi (0-9)
    'sql_kw_count', 'or_tautology', 'and_tautology', 'line_comment',
    'hash_comment', 'stmt_separator', 'sq_count', 'dq_count',
    'unbalanced_quotes', 'sleep_call',
    # XSS (10-20)
    'script_open', 'script_close', 'js_protocol', 'eval_call',
    'document_ref', 'event_handler_count', 'html_entity', 'url_enc_lt',
    'url_enc_gt', 'alert_call', 'dom_sink',
    # DirTrav (21-25)
    'dotdot_slash', 'dotdot_backslash', 'url_enc_dotdot', 'sensitive_path', 'trav_depth',
    # Log4Shell (26-33)
    'jndi_lookup', 'remote_scheme', 'nested_interp', 'env_lookup',
    'sys_lookup', 'jndi_in_header', 'rare_proto', 'obfuscated_jndi',
    # LogForging (34-38)
    'crlf_literal', 'crlf_url_enc', 'env_var_ref', 'cookie_jndi', 'log_level_token',
    # JNDI Placement (39-44)
    'jndi_in_url', 'jndi_in_body', 'jndi_in_hdr', 'sec_fetch_jndi',
    'header_only_jndi', 'multi_location_jndi',
    # Browser/Ctx (45-48)
    'mozilla_ua', 'sec_fetch_document', 'html_accept', 'forum_context',
    # Request Shape (49)
    'has_query_string',
]
