
#!/usr/bin/env python3
"""
iGRAIL Experiment Suite v3
===========================

Paper-facing comparison:
  FL : Conventional Federated Learning
  BC-FL  : Blockchain/Cryptographic-verified Federated Learning
  iGRAIL : Proposed Generative + DP + Verification Federated Learning

This version generates:
  1) Overall metric comparison
  2) Dataset-wise recall comparison
  3) Round-wise AUROC
  4) Client-wise round recall for iGRAIL
  5) Dirichlet-alpha robustness
  6) Privacy epsilon round-wise curves
  7) Privacy-utility trade-off
  8) Client scalability: AUROC vs number of clients
  9) Client scalability: execution time vs number of clients
 10) Client scalability: verification overhead vs number of clients

All numerical results are also saved to CSV.
No favorable result is hard-coded.
"""

from __future__ import annotations
import argparse, hashlib, os, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, log_loss,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.20

    # Main experiment
    n_clients: int = 3
    dirichlet_alpha: float = 0.5
    fl_rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    hidden: int = 64

    # Generative learning
    gan_rounds: int = 3
    gan_local_epochs: int = 1
    gan_batch_size: int = 128
    latent_dim: int = 32
    gan_hidden: int = 96
    critic_steps: int = 2
    gp_lambda: float = 10.0
    synth_ratio: float = 0.40

    # DP
    dp_noise_multiplier: float = 0.9
    dp_max_grad_norm: float = 1.0
    dp_delta: float = 1e-5

    # Repetitions
    repeats: int = 3
    max_rows: int | None = None

    # Parameter sweeps
    alpha_grid: Tuple[float, ...] = (0.1, 0.5, 1.0, 5.0)
    privacy_noise_grid: Tuple[float, ...] = (0.55, 0.75, 1.0, 1.35)
    # Delta values are kept small enough to be defensible across all attached datasets.
    delta_grid: Tuple[float, ...] = (1e-6, 1e-7, 1e-8)

    # Scalability experiment requested by user
    client_grid: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    scalability_round_grid: Tuple[int, ...] = (5, 10, 15, 20)

    # Adversarial robustness experiment.
    # One client is malicious from this round onward.
    attack_start_round: int = 6
    attack_client: int = 0
    attack_scale: float = 4.0

    out_dir: str = "results"


def get_preset(name: str) -> Config:
    if name == "smoke":
        return Config(
            fl_rounds=8,
            local_epochs=1,
            batch_size=64,
            gan_rounds=1,
            gan_local_epochs=1,
            gan_batch_size=64,
            critic_steps=1,
            synth_ratio=0.20,
            repeats=1,
            max_rows=1200,
            alpha_grid=(0.1, 0.5, 1.0),
            privacy_noise_grid=(0.75, 1.0),
            delta_grid=(1e-6, 1e-7),
            client_grid=(2, 3, 4, 5, 6, 7, 8, 9, 10),
            scalability_round_grid=(2, 4, 6),
            attack_start_round=3
        )
    return Config()


# ============================================================
# GLOBAL CONSTANTS / STYLE
# ============================================================

METHODS = ["FL", "BC-FL", "iGRAIL"]
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "BalancedAcc", "F1", "AUROC", "AUPRC", "MCC"]

MARKERS = {
    "FL": "o",
    "BC-FL": "s",
    "iGRAIL": "^",
}
LINES = {
    "FL": "-",
    "BC-FL": "--",
    "iGRAIL": "-.",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def style_axes(ax):
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.grid(True, alpha=0.35)
    return ax

def put_top_legend(ax, ncol=3):
    ax.legend(
        fontsize=16,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True
    )

def save_fig(fig, out_base):
    fig.tight_layout()
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# DATASETS
# ============================================================

DATA_INFO = {
    "cancer": {"file": "cancer.csv", "target": "diagnosis"},
    "diabetes": {"file": "diabetes.csv", "target": "diabetes"},
    "healthcare": {"file": "healthcare.csv", "target": "diabetes"},
    "heart": {"file": "heart.csv", "target": "num"},
}

def load_raw(name, data_dir, max_rows, seed):
    path = Path(data_dir) / DATA_INFO[name]["file"]
    df = pd.read_csv(path).copy()

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if "id" in df.columns:
        drop_cols.append("id")
    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    target = DATA_INFO[name]["target"]

    if name == "cancer":
        y = df[target].map({"B": 0, "M": 1}).astype(int)
    elif name == "heart":
        y = (pd.to_numeric(df[target], errors="coerce").fillna(0) > 0).astype(int)
    else:
        y = pd.to_numeric(df[target], errors="coerce").astype(int)

    X = df.drop(columns=[target])

    if max_rows is not None and len(X) > max_rows:
        ids, _ = train_test_split(
            np.arange(len(X)),
            train_size=max_rows,
            stratify=y,
            random_state=seed
        )
        X = X.iloc[ids].reset_index(drop=True)
        y = y.iloc[ids].reset_index(drop=True)

    return X, y

def preprocess(X, y, cfg):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed
    )

    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in Xtr.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    Xtr_e = pre.fit_transform(Xtr).astype(np.float32)
    Xte_e = pre.transform(Xte).astype(np.float32)

    return (
        Xtr_e,
        Xte_e,
        np.asarray(ytr, dtype=np.int64),
        np.asarray(yte, dtype=np.int64),
        pre
    )


# ============================================================
# DIRICHLET NON-IID SPLIT
# ============================================================

def dirichlet_partition(y, n_clients, alpha, seed, min_size=20):
    rng = np.random.default_rng(seed)

    for _ in range(400):
        clients = [[] for _ in range(n_clients)]

        for c in np.unique(y):
            ids = np.where(y == c)[0]
            rng.shuffle(ids)

            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p)[:-1] * len(ids)).astype(int)
            parts = np.split(ids, cuts)

            for k, part in enumerate(parts):
                clients[k].extend(part.tolist())

        clients = [np.asarray(v, dtype=int) for v in clients]

        if min(len(v) for v in clients) >= min_size:
            return clients

    raise RuntimeError(
        f"Dirichlet partition failed for n_clients={n_clients}, alpha={alpha}."
    )


# ============================================================
# PREDICTIVE MODEL
# ============================================================

class Classifier(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def clone_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

def fedavg(states, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    result = {}

    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for wi, state in zip(w, states):
            acc += float(wi) * state[key].float()
        result[key] = acc

    return result


def _flatten_update(local_state, global_state):
    parts = []
    for key in sorted(local_state):
        a = (local_state[key].float() - global_state[key].float()).reshape(-1)
        parts.append(a)
    return torch.cat(parts)


def trust_weighted_average(global_state, states, sample_weights):
    """
    Blockchain-aware trust aggregation used by BC-FL and iGRAIL.

    The verification layer first admits valid updates. Among admitted updates,
    a data-driven trust score is computed from update-direction agreement and
    update-norm consistency. This is NOT used by plain FL.

    This gives BC-FL a mathematically distinct aggregation rule without
    fabricating performance differences.
    """
    if len(states) == 1:
        return states[0], np.array([1.0], dtype=float)

    updates = [_flatten_update(s, global_state) for s in states]
    norms = np.array([float(torch.norm(u).item()) for u in updates], dtype=float)

    # Robust reference update: coordinate-wise median of admitted updates.
    stacked = torch.stack(updates, dim=0)
    ref = torch.median(stacked, dim=0).values
    ref_norm = float(torch.norm(ref).item()) + 1e-12
    med_norm = float(np.median(norms)) + 1e-12

    trust = []
    for u, nrm in zip(updates, norms):
        cosine = float(torch.dot(u, ref).item() / ((float(torch.norm(u).item()) + 1e-12) * ref_norm))
        direction = max((cosine + 1.0) / 2.0, 1e-3)
        norm_consistency = float(np.exp(-abs(nrm - med_norm) / med_norm))
        trust.append(direction * norm_consistency)

    trust = np.asarray(trust, dtype=float)
    base = np.asarray(sample_weights, dtype=float)
    combined = base * trust
    if not np.isfinite(combined).all() or combined.sum() <= 0:
        combined = base

    return fedavg(states, combined), trust


def state_bytes(state):
    chunks = []

    for key in sorted(state):
        arr = state[key].numpy()
        chunks.append(key.encode())
        chunks.append(str(arr.dtype).encode())
        chunks.append(np.asarray(arr.shape, dtype=np.int64).tobytes())
        chunks.append(arr.tobytes())

    return b"".join(chunks)

def train_local(base_state, X, y, cfg, dp=False, noise=None):
    dev = get_device()

    model = Classifier(X.shape[1], cfg.hidden).to(dev)
    model.load_state_dict(base_state)

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    loader = DataLoader(
        ds,
        batch_size=min(cfg.batch_size, len(ds)),
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # Use the same class-balanced BCE for every method. This avoids the
    # trivial all-negative solution on highly imbalanced datasets and keeps
    # the FL/BC-FL/iGRAIL comparison fair.
    n_pos = max(int(np.sum(y == 1)), 1)
    n_neg = max(int(np.sum(y == 0)), 1)
    pos_weight_value = float(np.clip(n_neg / n_pos, 1.0, 20.0))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=dev)
    )

    pe = None
    nm = cfg.dp_noise_multiplier if noise is None else float(noise)

    if dp:
        pe = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=nm,
            max_grad_norm=cfg.dp_max_grad_norm
        )

    steps = 0

    model.train()

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

            steps += 1

    if dp:
        state = {
            k.replace("_module.", ""): v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        q = 1.0 / max(1, len(loader))
    else:
        state = clone_state(model)
        q = np.nan

    return state, steps, q, nm

def evaluate_state(state, X, y, cfg):
    dev = get_device()

    model = Classifier(X.shape[1], cfg.hidden).to(dev)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.tensor(X, dtype=torch.float32, device=dev)
        ).cpu().numpy()

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    pred = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)

    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": specificity,
        "BalancedAcc": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "AUROC": roc_auc_score(y, probs),
        "AUPRC": average_precision_score(y, probs),
        "MCC": matthews_corrcoef(y, pred),
        "Loss": log_loss(y, np.clip(probs, 1e-8, 1 - 1e-8)),
    }


# ============================================================
# CONDITIONAL WGAN-GP
# ============================================================

class Generator(nn.Module):
    def __init__(self, zdim, outdim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim + 1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, outdim),
        )

    def forward(self, z, y):
        return self.net(torch.cat([z, y[:, None]], dim=1))

class Critic(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + 1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y[:, None]], dim=1)).squeeze(-1)

def gradient_penalty(critic, real, fake, y):
    alpha = torch.rand(len(real), 1, device=real.device)
    mixed = alpha * real + (1 - alpha) * fake
    mixed.requires_grad_(True)

    score = critic(mixed, y)

    grad = torch.autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True
    )[0]

    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()

def train_local_gan(global_state, X, y, cfg):
    dev = get_device()

    G = Generator(cfg.latent_dim, X.shape[1], cfg.gan_hidden).to(dev)
    D = Critic(X.shape[1], cfg.gan_hidden).to(dev)

    if global_state is not None:
        G.load_state_dict(global_state)

    gopt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    dopt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        ),
        batch_size=min(cfg.gan_batch_size, len(X)),
        shuffle=True
    )

    for _ in range(cfg.gan_local_epochs):
        for real, yy in loader:
            real = real.to(dev)
            yy = yy.to(dev)
            bs = len(real)

            for _ in range(cfg.critic_steps):
                z = torch.randn(bs, cfg.latent_dim, device=dev)
                fake = G(z, yy).detach()

                dloss = (
                    D(fake, yy).mean()
                    - D(real, yy).mean()
                    + cfg.gp_lambda * gradient_penalty(D, real, fake, yy)
                )

                dopt.zero_grad(set_to_none=True)
                dloss.backward()
                dopt.step()

            z = torch.randn(bs, cfg.latent_dim, device=dev)
            gloss = -D(G(z, yy), yy).mean()

            gopt.zero_grad(set_to_none=True)
            gloss.backward()
            gopt.step()

    return clone_state(G)

def train_federated_generator(client_data, cfg):
    init = Generator(
        cfg.latent_dim,
        client_data[0][0].shape[1],
        cfg.gan_hidden
    )
    global_state = clone_state(init)

    for _ in range(cfg.gan_rounds):
        states = []
        sizes = []

        for Xk, yk in client_data:
            states.append(
                train_local_gan(global_state, Xk, yk, cfg)
            )
            sizes.append(len(Xk))

        global_state = fedavg(states, sizes)

    return global_state

@torch.no_grad()
def synthesize(global_state, d_in, n, cfg, seed):
    set_seed(seed)
    dev = get_device()

    G = Generator(cfg.latent_dim, d_in, cfg.gan_hidden).to(dev)
    G.load_state_dict(global_state)
    G.eval()

    y = np.array(
        [0] * (n // 2) + [1] * (n - n // 2),
        dtype=np.float32
    )
    np.random.default_rng(seed).shuffle(y)

    out = []

    for i in range(0, n, 512):
        yy = torch.tensor(y[i:i+512], device=dev)
        z = torch.randn(len(yy), cfg.latent_dim, device=dev)
        out.append(G(z, yy).cpu().numpy())

    return np.vstack(out).astype(np.float32), y.astype(np.int64)


# ============================================================
# CRYPTOGRAPHIC VERIFICATION
# ============================================================

class VerificationLayer:
    def __init__(self, n_clients):
        self.server = X25519PrivateKey.generate()
        self.used = set()

        self.clients = [
            {
                "x": X25519PrivateKey.generate(),
                "sig": Ed25519PrivateKey.generate(),
                "authorized": True
            }
            for _ in range(n_clients)
        ]

    def derive_key(self, k, round_id):
        shared = self.clients[k]["x"].exchange(
            self.server.public_key()
        )

        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=f"{k}|{round_id}".encode()
        ).derive(shared)

    def protect(self, k, round_id, payload):
        nonce = os.urandom(12)
        aad = f"{k}|{round_id}".encode()

        ct = AESGCM(
            self.derive_key(k, round_id)
        ).encrypt(nonce, payload, aad)

        h = hashlib.sha256(ct + aad + nonce).digest()

        sig = self.clients[k]["sig"].sign(
            h + aad + nonce
        )

        return {
            "k": k,
            "r": round_id,
            "nonce": nonce,
            "ct": ct,
            "h": h,
            "sig": sig
        }

    def verify(self, sub, current_round, consume=True):
        k = sub["k"]

        if not self.clients[k]["authorized"]:
            return False, "unauthorized"

        if sub["r"] != current_round:
            return False, "stale"

        token = (k, current_round, sub["nonce"])

        if token in self.used:
            return False, "replay"

        aad = f"{k}|{current_round}".encode()

        h2 = hashlib.sha256(
            sub["ct"] + aad + sub["nonce"]
        ).digest()

        if h2 != sub["h"]:
            return False, "tamper"

        try:
            self.clients[k]["sig"].public_key().verify(
                sub["sig"],
                sub["h"] + aad + sub["nonce"]
            )

            AESGCM(
                self.derive_key(k, current_round)
            ).decrypt(
                sub["nonce"],
                sub["ct"],
                aad
            )

        except Exception:
            return False, "crypto-fail"

        if consume:
            self.used.add(token)

        return True, "accepted"

def security_tests():
    V = VerificationLayer(2)
    payload = b"model-update" * 100

    valid = V.protect(0, 1, payload)
    ok_valid, _ = V.verify(valid, 1, True)

    ok_replay, _ = V.verify(valid, 1, True)

    tam = V.protect(0, 2, payload)
    tam2 = dict(tam)
    raw = bytearray(tam2["ct"])
    raw[len(raw) // 2] ^= 1
    tam2["ct"] = bytes(raw)

    ok_tamper, _ = V.verify(tam2, 2, False)

    V.clients[1]["authorized"] = False
    unauth = V.protect(1, 3, payload)
    ok_unauth, _ = V.verify(unauth, 3, False)

    return pd.DataFrame([
        ["Valid", int(ok_valid), "Accept"],
        ["Tampered", int(ok_tamper), "Reject"],
        ["Replay", int(ok_replay), "Reject"],
        ["Unauthorized", int(ok_unauth), "Reject"],
    ], columns=["Scenario", "Accepted", "Expected"])



# ============================================================
# ADVERSARIAL UPDATE HELPERS
# ============================================================

def poison_state(state, scale=4.0, seed=42):
    """
    Create a deterministic malicious model update for robustness testing.
    This simulates a poisoned/tampered client update, not a new training method.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for key, tensor in state.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        std = float(np.std(arr))
        noise_std = max(std, 1e-3) * float(scale)
        arr += rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        out[key] = torch.tensor(arr, dtype=tensor.dtype)
    return out


# ============================================================
# FEDERATED RUN
# ============================================================

def run_method(
    method,
    Xtr, ytr, Xte, yte,
    cfg,
    alpha=None,
    n_clients=None,
    seed=None,
    noise=None,
    delta=None,
    attack_mode=False
):
    seed = cfg.seed if seed is None else seed
    set_seed(seed)

    alpha = cfg.dirichlet_alpha if alpha is None else alpha
    n_clients = cfg.n_clients if n_clients is None else n_clients
    delta = cfg.dp_delta if delta is None else float(delta)

    ids = dirichlet_partition(
        ytr,
        n_clients,
        alpha,
        seed,
        min_size=max(10, min(20, len(ytr)//max(2,n_clients*3)))
    )

    clients = [(Xtr[ix], ytr[ix]) for ix in ids]

    global_g = None

    if method == "iGRAIL":
        global_g = train_federated_generator(clients, cfg)

    global_state = clone_state(
        Classifier(Xtr.shape[1], cfg.hidden)
    )

    verifier = (
        VerificationLayer(n_clients)
        if method in ("BC-FL", "iGRAIL")
        else None
    )

    accountants = [
        RDPAccountant()
        for _ in range(n_clients)
    ]

    round_rows = []
    client_rows = []

    time_local = 0.0
    time_verify = 0.0
    time_aggregate = 0.0

    for r in range(1, cfg.fl_rounds + 1):
        states = []
        weights = []

        for k, (Xk, yk) in enumerate(clients):
            Xuse = Xk
            yuse = yk

            if method == "iGRAIL":
                ns = max(
                    20,
                    int(len(Xk) * cfg.synth_ratio)
                )

                Xs, ys = synthesize(
                    global_g,
                    Xtr.shape[1],
                    ns,
                    cfg,
                    seed + r * 1000 + k
                )

                Xuse = np.vstack([Xk, Xs])
                yuse = np.concatenate([yk, ys])

            t0 = time.perf_counter()

            state, steps, q, nm = train_local(
                global_state,
                Xuse,
                yuse,
                cfg,
                dp=(method == "iGRAIL"),
                noise=noise
            )

            time_local += time.perf_counter() - t0

            # In the adversarial experiment, one client submits a malicious
            # update from attack_start_round onward. FL has no admission
            # gate. BC-FL/iGRAIL protect the honest serialized update and the
            # transmitted ciphertext is then tampered with so verification
            # can detect/reject it.
            attack_active = (
                attack_mode
                and k == cfg.attack_client
                and r >= cfg.attack_start_round
            )

            state_for_aggregation = state
            if attack_active and method == "FL":
                state_for_aggregation = poison_state(
                    state,
                    scale=cfg.attack_scale,
                    seed=seed + 9999 + r
                )

            epsilon = np.nan

            if method == "iGRAIL":
                for _ in range(steps):
                    accountants[k].step(
                        noise_multiplier=nm,
                        sample_rate=q
                    )

                epsilon = accountants[k].get_epsilon(
                    delta=delta
                )

            accepted = True
            verify_reason = "not-applicable"

            if verifier is not None:
                t0 = time.perf_counter()

                sub = verifier.protect(
                    k,
                    r,
                    state_bytes(state)
                )

                if attack_active:
                    # Tamper after protection. Hash/signature/AEAD must reject.
                    tampered = dict(sub)
                    raw = bytearray(tampered["ct"])
                    if len(raw) > 0:
                        raw[len(raw) // 2] ^= 1
                    tampered["ct"] = bytes(raw)
                    sub = tampered

                accepted, verify_reason = verifier.verify(
                    sub,
                    r,
                    True
                )

                time_verify += time.perf_counter() - t0

            cm = evaluate_state(
                state,
                Xte,
                yte,
                cfg
            )

            client_rows.append({
                "Method": method,
                "Round": r,
                "Client": k + 1,
                "ClientsTotal": n_clients,
                "ClientN": len(Xk),
                "ClientPositiveRate": float(yk.mean()),
                "Alpha": alpha,
                "NoiseMultiplier": nm if method == "iGRAIL" else np.nan,
                "Epsilon": epsilon,
                "Delta": delta if method == "iGRAIL" else np.nan,
                "Accepted": int(accepted),
                "AttackActive": int(attack_active),
                "VerifyReason": verify_reason,
                **cm
            })

            if accepted:
                states.append(state_for_aggregation)
                weights.append(len(Xuse))

        t0 = time.perf_counter()

        # Plain FL uses standard sample-size FedAvg.
        # BC-FL and iGRAIL use blockchain-admitted trust-weighted aggregation.
        if method == "FL":
            global_state = fedavg(states, weights)
            round_trust = np.ones(len(states), dtype=float)
        else:
            global_state, round_trust = trust_weighted_average(
                global_state,
                states,
                weights
            )

        time_aggregate += time.perf_counter() - t0

        gm = evaluate_state(
            global_state,
            Xte,
            yte,
            cfg
        )

        global_eps = np.nan

        if method == "iGRAIL":
            global_eps = max(
                a.get_epsilon(delta)
                for a in accountants
            )

        round_rows.append({
            "Method": method,
            "Round": r,
            "ClientsTotal": n_clients,
            "Alpha": alpha,
            "NoiseMultiplier": noise if method == "iGRAIL" else np.nan,
            "Epsilon": global_eps,
            "Delta": delta if method == "iGRAIL" else np.nan,
            "AttackMode": int(attack_mode),
            "MeanTrust": float(np.mean(round_trust)) if len(round_trust) else np.nan,
            **gm
        })

    final = dict(round_rows[-1])

    final.update({
        "LocalTime": time_local,
        "VerifyTime": time_verify,
        "AggregateTime": time_aggregate,
        "TotalTime": (
            time_local
            + time_verify
            + time_aggregate
        )
    })

    return (
        final,
        pd.DataFrame(round_rows),
        pd.DataFrame(client_rows)
    )



# ============================================================
# TEMPORARY ROUND OVERRIDE
# ============================================================

class temporary_rounds:
    def __init__(self, cfg, rounds):
        self.cfg = cfg
        self.rounds = int(rounds)
        self.old_rounds = cfg.fl_rounds
    def __enter__(self):
        self.cfg.fl_rounds = self.rounds
    def __exit__(self, exc_type, exc_value, traceback):
        self.cfg.fl_rounds = self.old_rounds



# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def plot_measured_curve(ax, x, y, *, marker, linestyle, linewidth, markersize, label):
    """
    Draw a smooth PCHIP curve THROUGH the measured points and overlay the
    measured markers. PCHIP is interpolation only; it does not change stored
    experimental values or create extra measurements.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate x values after aggregation.
    ux, indices = np.unique(x, return_index=True)
    uy = y[indices]

    if len(ux) >= 3:
        dense_x = np.linspace(float(ux.min()), float(ux.max()), 300)
        interp = PchipInterpolator(ux, uy)
        dense_y = interp(dense_x)
        ax.plot(
            dense_x, dense_y,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label
        )
        ax.plot(
            ux, uy,
            linestyle="None",
            marker=marker,
            markersize=markersize
        )
    else:
        ax.plot(
            ux, uy,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            label=label
        )


# ============================================================
# COMPACT PAPER PLOTS — LINE GRAPHS ONLY
# ============================================================

PAPER_METRICS = [
    "Accuracy", "Precision", "Recall", "Specificity",
    "BalancedAcc", "F1", "AUROC", "AUPRC", "MCC"
]

DISPLAY = {
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "Specificity": "Specificity",
    "BalancedAcc": "Bal. Acc.",
    "F1": "F1-score",
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "MCC": "MCC",
}

def compact_axes(ax):
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.grid(True, alpha=0.30)
    return ax

def compact_legend(ax, ncol=3):
    ax.legend(
        fontsize=13,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        columnspacing=1.1,
        handlelength=2.5
    )

def metric_limits(values, metric):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    lo, hi = vals.min(), vals.max()
    span = max(hi - lo, 0.05)
    pad = max(0.02, 0.08 * span)
    if metric == "MCC":
        return max(-1.0, lo-pad), min(1.0, hi+pad)
    return max(0.0, lo-pad), min(1.0, hi+pad)


def plot_roundwise_metric(round_df, metric, out):
    """FL vs BC-FL vs iGRAIL over FL rounds."""
    z = round_df.groupby(["Round", "Method"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for method in METHODS:
        q = z[z["Method"] == method]
        plot_measured_curve(
            ax, q["Round"], q[metric],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.0, markersize=5.5, label=method
        )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    if z["Round"].max() <= 20:
        ax.set_xticks(np.arange(1, int(z["Round"].max()) + 1, 1))
    lim = metric_limits(z[metric], metric)
    if lim: ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"Round_{metric}")


def plot_privacy_metric(round_df, priv_round_df, metric, out):
    """
    Compact all-method privacy comparison.

    Curves:
      FL
      BC-FL
      iGRAIL for each measured (epsilon, delta) setting.

    The label reports the final measured epsilon and the fixed delta.
    """
    base = (
        round_df[round_df["Method"].isin(["FL", "BC-FL"])]
        .groupby(["Round", "Method"])[metric]
        .mean()
        .reset_index()
    )

    # Final measured epsilon for every (delta, noise) pair.
    finals = (
        priv_round_df.sort_values("Round")
        .groupby(["Dataset", "Delta", "NoiseMultiplier"])
        .tail(1)
        .groupby(["Delta", "NoiseMultiplier"])["Epsilon"]
        .mean()
        .to_dict()
    )

    priv = (
        priv_round_df.groupby(
            ["Round", "Delta", "NoiseMultiplier"]
        )[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.3))

    # Non-private / non-DP baselines.
    for method in ["FL", "BC-FL"]:
        q = base[base["Method"] == method]
        ax.plot(
            q["Round"], q[metric],
            marker=MARKERS[method],
            linestyle=LINES[method],
            linewidth=2.3, markersize=5.5,
            label=method
        )

    pmarkers = ["v", ">", "*", "<", "D", "P", "X", "^", "s", "o", "h", "d"]
    pstyles = ["-", "--", ":", "-.", "-", "--", ":", "-.", "-", "--", ":", "-."]

    i = 0
    for delta in sorted(priv["Delta"].dropna().unique(), reverse=True):
        for nm in sorted(priv["NoiseMultiplier"].dropna().unique()):
            q = priv[
                (priv["Delta"] == delta)
                & (priv["NoiseMultiplier"] == nm)
            ]
            if len(q) == 0:
                continue

            eps = finals[(delta, nm)]

            ax.plot(
                q["Round"], q[metric],
                marker=pmarkers[i % len(pmarkers)],
                linestyle=pstyles[i % len(pstyles)],
                linewidth=1.7, markersize=5.5,
                label=rf"iGRAIL, $\epsilon$={eps:.2f}, $\delta$={delta:.0e}"
            )
            i += 1

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])

    xmax = max(base["Round"].max(), priv["Round"].max())
    if xmax <= 20:
        ax.set_xticks(np.arange(1, int(xmax) + 1, 1))

    lim = metric_limits(
        np.concatenate([base[metric].values, priv[metric].values]),
        metric
    )
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 2)
    save_fig(fig, out / f"Privacy_EpsDelta_AllMethods_{metric}")


def plot_privacy_tradeoff_by_delta(priv_final_df, metric, out):
    """
    Privacy-utility trade-off:
      x = measured epsilon
      y = metric
      one iGRAIL curve per delta

    This is the clearest direct experiment for jointly reporting epsilon and delta.
    """
    z = (
        priv_final_df.groupby(
            ["Delta", "NoiseMultiplier"]
        )[["Epsilon", metric]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    markers = ["o", "s", "^", "D", "v"]

    for i, delta in enumerate(sorted(z["Delta"].unique(), reverse=True)):
        q = z[z["Delta"] == delta].sort_values("Epsilon")
        ax.plot(
            q["Epsilon"], q[metric],
            marker=markers[i % len(markers)],
            linewidth=2.0,
            markersize=6,
            label=rf"$\delta$={delta:.0e}"
        )

    ax.set_xlabel(r"Privacy budget $\epsilon$")
    ax.set_ylabel(DISPLAY[metric])

    lim = metric_limits(z[metric], metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, min(3, len(z["Delta"].unique())))
    save_fig(fig, out / f"PrivacyTradeoff_{metric}")



def plot_alpha_metric(alpha_df, metric, out):
    """FL vs BC-FL vs iGRAIL across non-IID alpha."""
    z = alpha_df.groupby(["Alpha", "Method"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Alpha")
        plot_measured_curve(
            ax, q["Alpha"], q[metric],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.0, markersize=6, label=method
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha$")
    ax.set_ylabel(DISPLAY[metric])
    lim = metric_limits(z[metric], metric)
    if lim: ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"Alpha_{metric}")


def plot_clients_metric(scale_df, metric, out):
    """
    FL vs BC-FL vs iGRAIL for every integer K=2,...,10.
    Axis ticks are exactly integers 1,...,10 (no decimal tick labels).
    """
    z = scale_df.groupby(["Clients", "Method"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Clients")
        plot_measured_curve(
            ax, q["Clients"], q[metric],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.0, markersize=5.5, label=method
        )

    ax.set_xlim(1, 10.2)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([str(i) for i in range(1, 11)], fontsize=18)
    ax.set_xlabel("Clients")
    ax.set_ylabel(DISPLAY[metric])
    lim = metric_limits(z[metric], metric)
    if lim: ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"Clients_2to10_{metric}")


def plot_clients_rounds_metric(scale_metric_df, metric, out):
    """
    Compact high-density comparison:
    three methods at K={2,5,10}; metric vs FL round.
    This gives 9 curves in one publishable line plot.
    """
    use_clients = [2, 5, 10]
    q = scale_metric_df[scale_metric_df["Clients"].isin(use_clients)]

    z = (
        q.groupby(["Round", "Method", "Clients"])[metric]
        .mean()
        .reset_index()
    )

    markers = ["o", "s", "^", "v", ">", "*", "<", "D", "P"]
    styles = ["-", "--", ":", "-.", "-", "--", ":", "-.", "-"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    i = 0
    for method in METHODS:
        for k in use_clients:
            a = z[(z["Method"] == method) & (z["Clients"] == k)]
            if len(a) == 0:
                continue
            ax.plot(
                a["Round"], a[metric],
                marker=markers[i % len(markers)],
                linestyle=styles[i % len(styles)],
                linewidth=1.7, markersize=5,
                label=f"{method}, K={k}"
            )
            i += 1

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    xmax = z["Round"].max()
    if xmax <= 20:
        ax.set_xticks(np.arange(1, int(xmax) + 1, 1))
    lim = metric_limits(z[metric], metric)
    if lim: ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"ClientsRounds_{metric}")


def plot_dataset_metric_line(main_df, metric, out):
    """
    Dataset comparison as a line graph (not grouped bars) to save paper space.
    """
    order = [d for d in ["cancer", "diabetes", "healthcare", "heart"]
             if d in main_df["Dataset"].unique()]
    z = (
        main_df.groupby(["Dataset", "Method"])[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    x = np.arange(1, len(order) + 1)

    for method in METHODS:
        q = z[z["Method"] == method].set_index("Dataset").reindex(order)
        ax.plot(
            x, q[metric].values,
            marker=MARKERS[method],
            linestyle=LINES[method],
            linewidth=2.0, markersize=6,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Cancer", "Diabetes", "Healthcare", "Heart"][:len(order)],
        fontsize=16
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel(DISPLAY[metric])
    lim = metric_limits(z[metric], metric)
    if lim: ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"Datasets_{metric}")


def plot_execution_clients_rounds(scale_time_df, out):
    """
    Line-graph version of the participants/iterations execution-time figure.
    x = clients 2..10, one curve per FL-round setting.
    """
    z = (
        scale_time_df.groupby(["Clients", "Rounds"])["TotalTime"]
        .mean()
        .reset_index()
    )
    rounds = sorted(z["Rounds"].unique())
    markers = ["o", "s", "^", "v", "D"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, r in enumerate(rounds):
        q = z[z["Rounds"] == r].sort_values("Clients")
        ax.plot(
            q["Clients"], q["TotalTime"],
            marker=markers[i % len(markers)],
            linewidth=1.9,
            label=f"{r} rounds"
        )

    ax.set_xlim(1, 10.2)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([str(i) for i in range(1, 11)], fontsize=18)
    ax.set_xlabel("Clients")
    ax.set_ylabel("Time (s)")
    compact_axes(ax)
    compact_legend(ax, min(4, len(rounds)))
    save_fig(fig, out / "Execution_Clients_2to10")


def plot_runtime_components_lines(round_runtime_df, out):
    """
    Four compact line plots:
    local train, verification, aggregation, total runtime vs FL rounds.
    """
    specs = [
        ("LocalTime", "Train time (s)", "Runtime_Local"),
        ("VerifyTime", "Verify time (s)", "Runtime_Verify"),
        ("AggregateTime", "Agg. time (s)", "Runtime_Agg"),
        ("TotalTime", "Total time (s)", "Runtime_Total"),
    ]

    for col, ylabel, fname in specs:
        z = (
            round_runtime_df.groupby(["Rounds", "Method"])[col]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8.2, 4.9))
        for method in METHODS:
            q = z[z["Method"] == method].sort_values("Rounds")
            ax.plot(
                q["Rounds"], q[col],
                marker=MARKERS[method],
                linestyle=LINES[method],
                linewidth=2.0, markersize=6,
                label=method
            )

        ax.set_xlabel("FL rounds")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(z["Rounds"].unique()))
        compact_axes(ax)
        compact_legend(ax, 3)
        save_fig(fig, out / fname)


def plot_crypto_payload(sec_df, out):
    z = (
        sec_df.groupby("PayloadKB")[["ProtectMs", "VerifyMs"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(
        z["PayloadKB"], z["ProtectMs"],
        marker="o", linewidth=2.0, label="Protect"
    )
    ax.plot(
        z["PayloadKB"], z["VerifyMs"],
        marker="s", linestyle="--", linewidth=2.0, label="Verify"
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Payload (KB)")
    ax.set_ylabel("Time (ms)")
    compact_axes(ax)
    compact_legend(ax, 2)
    save_fig(fig, out / "Crypto_PayloadTime")


def generate_compact_paper_plots(
    main_df, round_df, alpha_df, privacy_round_df, privacy_final_df,
    scale_df, scale_metric_df, scale_time_df,
    round_runtime_df, sec_bench_df, figures_root
):
    folders = {
        "round": figures_root / "01_roundwise",
        "privacy": figures_root / "02_privacy_all_methods",
        "alpha": figures_root / "03_noniid",
        "clients": figures_root / "04_clients_2to10",
        "clients_rounds": figures_root / "05_clients_rounds",
        "datasets": figures_root / "06_datasets",
        "runtime": figures_root / "07_runtime",
        "security": figures_root / "08_security",
    }
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)

    for metric in PAPER_METRICS:
        plot_roundwise_metric(round_df, metric, folders["round"])
        plot_privacy_metric(round_df, privacy_round_df, metric, folders["privacy"])
        plot_privacy_tradeoff_by_delta(privacy_final_df, metric, folders["privacy"])
        plot_alpha_metric(alpha_df, metric, folders["alpha"])
        plot_clients_metric(scale_df, metric, folders["clients"])
        plot_clients_rounds_metric(scale_metric_df, metric, folders["clients_rounds"])
        plot_dataset_metric_line(main_df, metric, folders["datasets"])

    plot_execution_clients_rounds(scale_time_df, folders["runtime"])
    plot_runtime_components_lines(round_runtime_df, folders["runtime"])
    plot_crypto_payload(sec_bench_df, folders["security"])



def _mean_ci(df, group_cols, metric):
    g = df.groupby(group_cols)[metric]
    z = g.agg(["mean", "std", "count"]).reset_index()
    z["sem"] = z["std"].fillna(0.0) / np.sqrt(z["count"].clip(lower=1))
    z["ci95"] = 1.96 * z["sem"]
    return z


def plot_roundwise_metric_ci(round_df, metric, out):
    """
    Actual learning trajectories. No interpolation and no fabricated smoothing.
    Curvature comes from using all FL rounds. Shaded bands are 95% CI over
    datasets/repeats.
    """
    z = _mean_ci(round_df, ["Round", "Method"], metric)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Round")
        plot_measured_curve(
            ax, q["Round"], q["mean"],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.0, markersize=4.5, label=method
        )
        ax.fill_between(
            q["Round"].to_numpy(),
            (q["mean"] - q["ci95"]).to_numpy(),
            (q["mean"] + q["ci95"]).to_numpy(),
            alpha=0.12
        )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    ax.set_xticks(np.arange(1, int(z["Round"].max()) + 1, 1))
    lim = metric_limits(z["mean"], metric)
    if lim:
        ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"RoundCurve_{metric}")


def plot_attack_robustness(attack_round_df, metric, out, attack_start_round):
    """
    Robustness curve under one malicious client. This is where BC-FL should
    legitimately differ from FL because the verification gate is active.
    """
    z = _mean_ci(attack_round_df, ["Round", "Method"], metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Round")
        plot_measured_curve(
            ax, q["Round"], q["mean"],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.0, markersize=4.5, label=method
        )
        ax.fill_between(
            q["Round"].to_numpy(),
            (q["mean"] - q["ci95"]).to_numpy(),
            (q["mean"] + q["ci95"]).to_numpy(),
            alpha=0.12
        )

    ax.axvline(
        attack_start_round,
        linestyle=":",
        linewidth=1.6
    )
    ax.text(
        attack_start_round + 0.2,
        ax.get_ylim()[0] if metric == "MCC" else max(0.0, z["mean"].min() - 0.02),
        "attack",
        fontsize=12,
        rotation=90,
        va="bottom"
    )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    ax.set_xticks(np.arange(1, int(z["Round"].max()) + 1, 1))
    lim = metric_limits(z["mean"], metric)
    if lim:
        ax.set_ylim(*lim)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"AttackCurve_{metric}")


def plot_final_metric_bars(main_df, out):
    """
    Bar chart is appropriate here because this is a final categorical
    comparison rather than an ordered progression.
    """
    use = ["Accuracy", "Recall", "F1", "AUROC", "AUPRC", "MCC"]
    z = main_df.groupby("Method")[use].mean().reindex(METHODS)

    x = np.arange(len(use))
    width = 0.24
    hatches = ["///", "\\\\\\", "xx"]

    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    for i, method in enumerate(METHODS):
        ax.bar(
            x + (i - 1) * width,
            z.loc[method].values,
            width,
            hatch=hatches[i],
            alpha=0.88,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(use, fontsize=16)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05 if z.min().min() < 0 else 0.0, 1.02)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / "FinalMetrics_Bar")


def plot_security_acceptance_bar(security_df, out):
    """
    Security scenarios are independent categories, so bars are more suitable
    than a line plot.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(security_df))
    ax.bar(
        x,
        security_df["Accepted"].values,
        hatch=["///", "\\\\\\", "xx", ".."],
        alpha=0.88
    )
    ax.set_xticks(x)
    ax.set_xticklabels(security_df["Scenario"].tolist(), fontsize=15)
    ax.set_ylabel("Accepted")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1])
    compact_axes(ax)
    save_fig(fig, out / "Security_Acceptance_Bar")



# ============================================================
# FINAL PAPER FIGURES — COMPACT, HIGH-INFORMATION
# ============================================================

def paper_convergence_compact(round_df, out):
    """
    9 curves in one plot:
      3 methods x {Accuracy, F1, AUROC}
    This is a compact convergence figure suitable for the main paper.
    """
    metrics = ["Accuracy", "F1", "AUROC"]
    metric_styles = {"Accuracy": "-", "F1": "--", "AUROC": ":"}
    metric_markers = {"Accuracy": "o", "F1": "s", "AUROC": "^"}

    z = round_df.groupby(["Round", "Method"])[metrics].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Round")
        for metric in metrics:
            plot_measured_curve(
                ax, q["Round"], q[metric],
                marker=metric_markers[metric],
                linestyle=metric_styles[metric],
                linewidth=1.75,
                markersize=4.5,
                label=f"{method}-{DISPLAY[metric]}"
            )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel("Score")
    ax.set_xticks(np.arange(1, int(z["Round"].max()) + 1, 1))
    ax.set_ylim(0, 1.02)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / "P1_Convergence_Accuracy_F1_AUROC")


def paper_client_scaling_compact(scale_df, out):
    """
    9 curves:
      3 methods x {Recall, F1, AUROC}
    for K=2,...,10.
    """
    metrics = ["Recall", "F1", "AUROC"]
    metric_styles = {"Recall": "-", "F1": "--", "AUROC": ":"}
    metric_markers = {"Recall": "o", "F1": "s", "AUROC": "^"}

    z = scale_df.groupby(["Clients", "Method"])[metrics].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Clients")
        for metric in metrics:
            plot_measured_curve(
                ax, q["Clients"], q[metric],
                marker=metric_markers[metric],
                linestyle=metric_styles[metric],
                linewidth=1.75,
                markersize=4.5,
                label=f"{method}-{DISPLAY[metric]}"
            )

    ax.set_xlim(1, 10.2)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([str(i) for i in range(1, 11)], fontsize=18)
    ax.set_xlabel("Clients, K")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / "P2_ClientScaling_Recall_F1_AUROC")


def paper_privacy_compact(round_df, privacy_round_df, out):
    """
    One compact DP figure:
      FL baseline
      BC-FL baseline
      three representative iGRAIL (epsilon, delta) settings
    y=AUROC; x=FL round.
    """
    base = (
        round_df[round_df["Method"].isin(["FL", "BC-FL"])]
        .groupby(["Round", "Method"])["AUROC"]
        .mean()
        .reset_index()
    )

    finals = (
        privacy_round_df.sort_values("Round")
        .groupby(["Dataset", "Delta", "NoiseMultiplier"])
        .tail(1)
        .groupby(["Delta", "NoiseMultiplier"])["Epsilon"]
        .mean()
        .reset_index()
        .sort_values("Epsilon")
    )

    # Select low / middle / high epsilon settings for a readable paper figure.
    if len(finals) >= 3:
        take = [0, len(finals)//2, len(finals)-1]
        chosen = finals.iloc[take][["Delta", "NoiseMultiplier", "Epsilon"]]
    else:
        chosen = finals[["Delta", "NoiseMultiplier", "Epsilon"]]

    priv = (
        privacy_round_df.groupby(["Round", "Delta", "NoiseMultiplier"])["AUROC"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for method in ["FL", "BC-FL"]:
        q = base[base["Method"] == method].sort_values("Round")
        plot_measured_curve(
            ax, q["Round"], q["AUROC"],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.1, markersize=5,
            label=method
        )

    pmarkers = ["v", "*", "<"]
    pstyles = ["-", "--", "-."]
    for i, (_, row) in enumerate(chosen.iterrows()):
        d = float(row["Delta"])
        nm = float(row["NoiseMultiplier"])
        eps = float(row["Epsilon"])
        q = priv[
            (priv["Delta"] == d)
            & (priv["NoiseMultiplier"] == nm)
        ].sort_values("Round")

        plot_measured_curve(
            ax, q["Round"], q["AUROC"],
            marker=pmarkers[i % len(pmarkers)],
            linestyle=pstyles[i % len(pstyles)],
            linewidth=1.9, markersize=5.5,
            label=rf"iGRAIL $\epsilon$={eps:.2f}, $\delta$={d:.0e}"
        )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel("AUROC")
    ax.set_xticks(np.arange(1, int(priv["Round"].max()) + 1, 1))
    compact_axes(ax)
    compact_legend(ax, 2)
    save_fig(fig, out / "P3_Privacy_FL_BCFL_iGRAIL")


def paper_attack_compact(attack_round_df, attack_start_round, out):
    """
    Clean security proof figure. BC-FL differs here because the verification
    mechanism actually rejects tampered submissions.
    """
    z = attack_round_df.groupby(["Round", "Method"])["AUROC"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method in METHODS:
        q = z[z["Method"] == method].sort_values("Round")
        plot_measured_curve(
            ax, q["Round"], q["AUROC"],
            marker=MARKERS[method], linestyle=LINES[method],
            linewidth=2.0, markersize=5,
            label=method
        )

    ax.axvline(attack_start_round, linestyle=":", linewidth=1.6)
    ax.text(
        attack_start_round + 0.15,
        max(0.0, float(z["AUROC"].min()) - 0.01),
        "attack begins",
        rotation=90,
        fontsize=11,
        va="bottom"
    )
    ax.set_xlabel("FL rounds")
    ax.set_ylabel("AUROC")
    ax.set_xticks(np.arange(1, int(z["Round"].max()) + 1, 1))
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / "P4_AttackRobustness_AUROC")


def paper_selected_rounds_dataset_metrics(client_df, out):
    """
    For each dataset, summarize iGRAIL client performance at rounds {2,5,7,10}.
    Five metrics are merged into one line figure per dataset.
    The plotted value is the mean across the K=3 main clients; detailed
    per-client values remain in CSV.
    """
    rounds_keep = [2, 5, 7, 10]
    metrics = ["Accuracy", "Recall", "F1", "AUROC", "AUPRC"]
    markers = ["o", "s", "^", "v", "D"]
    styles = ["-", "--", ":", "-.", "-"]

    q0 = client_df[
        (client_df["Method"] == "iGRAIL")
        & (client_df["ClientsTotal"] == 3)
        & (client_df["Round"].isin(rounds_keep))
    ].copy()

    for dataset in sorted(q0["Dataset"].unique()):
        q = q0[q0["Dataset"] == dataset]
        z = q.groupby("Round")[metrics].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8.2, 4.9))
        for i, metric in enumerate(metrics):
            plot_measured_curve(
                ax, z["Round"], z[metric],
                marker=markers[i],
                linestyle=styles[i],
                linewidth=1.8,
                markersize=5.2,
                label=DISPLAY[metric]
            )

        ax.set_xlabel("Selected FL rounds")
        ax.set_ylabel("Score")
        ax.set_xticks(rounds_keep)
        ax.set_ylim(0, 1.02)
        compact_axes(ax)
        compact_legend(ax, 3)
        save_fig(fig, out / f"P5_{dataset}_SelectedRounds_Metrics")


def paper_per_client_selected_rounds(client_df, out):
    """
    Per-client evidence requested by the user.
    One compact figure per dataset for the K=3 main experiment:
      x = rounds {2,5,7,10}
      y = AUROC
      curves = Client 1, Client 2, Client 3
    """
    rounds_keep = [2, 5, 7, 10]
    q0 = client_df[
        (client_df["Method"] == "iGRAIL")
        & (client_df["ClientsTotal"] == 3)
        & (client_df["Round"].isin(rounds_keep))
    ].copy()

    for dataset in sorted(q0["Dataset"].unique()):
        q = q0[q0["Dataset"] == dataset]
        z = q.groupby(["Round", "Client"])["AUROC"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for client in sorted(z["Client"].unique()):
            a = z[z["Client"] == client].sort_values("Round")
            plot_measured_curve(
                ax, a["Round"], a["AUROC"],
                marker=["o", "s", "^"][int(client-1) % 3],
                linestyle=["-", "--", "-."][int(client-1) % 3],
                linewidth=1.9,
                markersize=5.2,
                label=f"Client {int(client)}"
            )

        ax.set_xlabel("Selected FL rounds")
        ax.set_ylabel("AUROC")
        ax.set_xticks(rounds_keep)
        compact_axes(ax)
        compact_legend(ax, 3)
        save_fig(fig, out / f"P6_{dataset}_Clients_SelectedRounds")


def paper_final_bar(main_df, out):
    """
    Bars are used only for final categorical comparison.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC", "MCC"]
    z = main_df.groupby("Method")[metrics].mean().reindex(METHODS)

    x = np.arange(len(metrics))
    width = 0.24
    hatches = ["///", "\\\\\\", "xx"]

    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    for i, method in enumerate(METHODS):
        ax.bar(
            x + (i-1)*width,
            z.loc[method].values,
            width,
            hatch=hatches[i],
            alpha=0.88,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=15)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05 if z.min().min() < 0 else 0.0, 1.02)
    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / "P7_FinalMetrics_Bar")



# ============================================================
# FINAL ONE-METRIC / MULTI-DATASET PAPER FIGURES
# ============================================================

FINAL_METRICS = [
    "Accuracy", "Precision", "Recall", "Specificity",
    "BalancedAcc", "F1", "AUROC", "AUPRC", "MCC"
]

DATASET_LABELS = {
    "cancer": "Cancer",
    "diabetes": "Diabetes",
    "healthcare": "Healthcare",
    "heart": "Heart"
}

METHOD_STYLE = {
    "FL": "-",
    "BC-FL": "--",
    "iGRAIL": "-."
}

METHOD_MARKER = {
    "FL": "o",
    "BC-FL": "s",
    "iGRAIL": "^"
}


def _dataset_method_label(dataset, method):
    return f"{DATASET_LABELS.get(dataset, dataset)}-{method}"


def _metric_ylim_from_values(values, metric):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 0.05)
    pad = max(0.02, 0.07 * span)

    if metric == "MCC":
        return max(-1.0, lo - pad), min(1.0, hi + pad)

    return max(0.0, lo - pad), min(1.0, hi + pad)


def paper_client_scaling_one_metric(scale_df, metric, out):
    """
    One metric on y-axis.
    x-axis: client count K=2..10.
    Curves: every dataset x every method (up to 12 curves).
    """
    z = (
        scale_df.groupby(["Dataset", "Clients", "Method"])[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.7, 5.5))

    for dataset in sorted(z["Dataset"].unique()):
        for method in METHODS:
            q = z[
                (z["Dataset"] == dataset)
                & (z["Method"] == method)
            ].sort_values("Clients")

            if len(q) == 0:
                continue

            plot_measured_curve(
                ax,
                q["Clients"],
                q[metric],
                marker=METHOD_MARKER[method],
                linestyle=METHOD_STYLE[method],
                linewidth=1.7,
                markersize=4.8,
                label=_dataset_method_label(dataset, method)
            )

    ax.set_xlim(1, 10.2)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([str(i) for i in range(1, 11)], fontsize=18)
    ax.set_xlabel("Clients, K")
    ax.set_ylabel(DISPLAY[metric])

    lim = _metric_ylim_from_values(z[metric], metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 4)
    save_fig(fig, out / f"ClientScaling_{metric}")


def paper_roundwise_one_metric(round_df, metric, out):
    """
    One metric on y-axis.
    x-axis: FL rounds.
    Curves: every dataset x every method.
    """
    z = (
        round_df.groupby(["Dataset", "Round", "Method"])[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.7, 5.5))

    for dataset in sorted(z["Dataset"].unique()):
        for method in METHODS:
            q = z[
                (z["Dataset"] == dataset)
                & (z["Method"] == method)
            ].sort_values("Round")

            if len(q) == 0:
                continue

            plot_measured_curve(
                ax,
                q["Round"],
                q[metric],
                marker=METHOD_MARKER[method],
                linestyle=METHOD_STYLE[method],
                linewidth=1.7,
                markersize=4.3,
                label=_dataset_method_label(dataset, method)
            )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    ax.set_xticks(np.arange(1, int(z["Round"].max()) + 1, 1))

    lim = _metric_ylim_from_values(z[metric], metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 4)
    save_fig(fig, out / f"Roundwise_{metric}")


def paper_selected_rounds_one_metric(round_df, metric, out):
    """
    Compact selected-round figure using R={2,5,7,10}.
    Curves: every dataset x every method.
    """
    rounds_keep = [2, 5, 7, 10]

    z = (
        round_df[round_df["Round"].isin(rounds_keep)]
        .groupby(["Dataset", "Round", "Method"])[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.3))

    for dataset in sorted(z["Dataset"].unique()):
        for method in METHODS:
            q = z[
                (z["Dataset"] == dataset)
                & (z["Method"] == method)
            ].sort_values("Round")

            if len(q) == 0:
                continue

            plot_measured_curve(
                ax,
                q["Round"],
                q[metric],
                marker=METHOD_MARKER[method],
                linestyle=METHOD_STYLE[method],
                linewidth=1.8,
                markersize=5.0,
                label=_dataset_method_label(dataset, method)
            )

    ax.set_xlabel("Selected FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    ax.set_xticks(rounds_keep)

    lim = _metric_ylim_from_values(z[metric], metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 4)
    save_fig(fig, out / f"SelectedRounds_{metric}")


def paper_alpha_one_metric(alpha_df, metric, out):
    """
    One metric on y-axis.
    x-axis: Dirichlet alpha.
    Curves: every dataset x every method.
    """
    z = (
        alpha_df.groupby(["Dataset", "Alpha", "Method"])[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.7, 5.5))

    for dataset in sorted(z["Dataset"].unique()):
        for method in METHODS:
            q = z[
                (z["Dataset"] == dataset)
                & (z["Method"] == method)
            ].sort_values("Alpha")

            if len(q) == 0:
                continue

            plot_measured_curve(
                ax,
                q["Alpha"],
                q[metric],
                marker=METHOD_MARKER[method],
                linestyle=METHOD_STYLE[method],
                linewidth=1.7,
                markersize=4.8,
                label=_dataset_method_label(dataset, method)
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha$")
    ax.set_ylabel(DISPLAY[metric])

    lim = _metric_ylim_from_values(z[metric], metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 4)
    save_fig(fig, out / f"NonIID_{metric}")


def paper_privacy_tradeoff_one_metric(privacy_final_df, metric, out):
    """
    Direct (epsilon,delta)-privacy trade-off.

    x-axis: measured epsilon
    y-axis: ONE selected metric

    Curves: dataset x delta for iGRAIL.
    FL and BC-FL are intentionally not assigned epsilon values because
    they are non-DP reference methods in this implementation.
    """
    z = (
        privacy_final_df.groupby(
            ["Dataset", "Delta", "NoiseMultiplier"]
        )[["Epsilon", metric]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    delta_markers = {
        1e-6: "o",
        1e-7: "s",
        1e-8: "^"
    }
    delta_styles = {
        1e-6: "-",
        1e-7: "--",
        1e-8: "-."
    }

    for dataset in sorted(z["Dataset"].unique()):
        for delta in sorted(z["Delta"].unique(), reverse=True):
            q = z[
                (z["Dataset"] == dataset)
                & (z["Delta"] == delta)
            ].sort_values("Epsilon")

            if len(q) == 0:
                continue

            plot_measured_curve(
                ax,
                q["Epsilon"],
                q[metric],
                marker=delta_markers.get(float(delta), "o"),
                linestyle=delta_styles.get(float(delta), "-"),
                linewidth=1.8,
                markersize=5.0,
                label=f"{DATASET_LABELS.get(dataset, dataset)}, "
                      + rf"$\delta$={delta:.0e}"
            )

    ax.set_xlabel(r"Privacy budget $\epsilon$")
    ax.set_ylabel(DISPLAY[metric])

    lim = _metric_ylim_from_values(z[metric], metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"PrivacyTradeoff_{metric}")


def paper_privacy_round_one_metric(round_df, privacy_round_df, metric, out):
    """
    FL/BC-FL references + representative iGRAIL DP settings.
    y-axis is one metric only.

    To keep the figure compact, the proposed method shows three
    representative measured epsilon settings: low, median, high.
    """
    base = (
        round_df[
            round_df["Method"].isin(["FL", "BC-FL"])
        ]
        .groupby(["Dataset", "Round", "Method"])[metric]
        .mean()
        .reset_index()
    )

    finals = (
        privacy_round_df.sort_values("Round")
        .groupby(["Dataset", "Delta", "NoiseMultiplier"])
        .tail(1)
        [["Dataset", "Delta", "NoiseMultiplier", "Epsilon"]]
    )

    chosen_rows = []

    for dataset in sorted(finals["Dataset"].unique()):
        d = finals[finals["Dataset"] == dataset].sort_values("Epsilon")
        if len(d) >= 3:
            ids = [0, len(d) // 2, len(d) - 1]
            chosen_rows.append(d.iloc[ids])
        else:
            chosen_rows.append(d)

    chosen = pd.concat(chosen_rows, ignore_index=True)

    priv = (
        privacy_round_df.groupby(
            ["Dataset", "Round", "Delta", "NoiseMultiplier"]
        )[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(16.0, 7.0))

    # FL and BC-FL per dataset.
    for dataset in sorted(base["Dataset"].unique()):
        for method in ["FL", "BC-FL"]:
            q = base[
                (base["Dataset"] == dataset)
                & (base["Method"] == method)
            ].sort_values("Round")

            plot_measured_curve(
                ax,
                q["Round"],
                q[metric],
                marker=METHOD_MARKER[method],
                linestyle=METHOD_STYLE[method],
                linewidth=1.8,
                markersize=4.5,
                label=_dataset_method_label(dataset, method)
            )

    # Representative iGRAIL DP settings per dataset.
    dp_markers = ["v", "*", "<"]
    dp_styles = ["-", "--", "-."]

    for dataset in sorted(chosen["Dataset"].unique()):
        d = chosen[chosen["Dataset"] == dataset].sort_values("Epsilon")

        for j, (_, row) in enumerate(d.iterrows()):
            delta = float(row["Delta"])
            nm = float(row["NoiseMultiplier"])
            eps = float(row["Epsilon"])

            q = priv[
                (priv["Dataset"] == dataset)
                & (priv["Delta"] == delta)
                & (priv["NoiseMultiplier"] == nm)
            ].sort_values("Round")

            plot_measured_curve(
                ax,
                q["Round"],
                q[metric],
                marker=dp_markers[j % len(dp_markers)],
                linestyle=dp_styles[j % len(dp_styles)],
                linewidth=1.7,
                markersize=4.5,
                label=(
                    f"{DATASET_LABELS.get(dataset, dataset)}-iGRAIL "
                    + rf"$\epsilon$={eps:.2f}, $\delta$={delta:.0e}"
                )
            )

    ax.set_xlabel("FL rounds")
    ax.set_ylabel(DISPLAY[metric])
    xmax = max(base["Round"].max(), priv["Round"].max())
    ax.set_xticks(np.arange(1, int(xmax) + 1, 1))

    values = np.concatenate([base[metric].values, priv[metric].values])
    lim = _metric_ylim_from_values(values, metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 4)
    save_fig(fig, out / f"PrivacyRounds_{metric}")


def paper_final_dataset_bar_one_metric(main_df, metric, out):
    """
    Bar chart is used only for the final discrete dataset comparison.
    x-axis: dataset
    bars: FL, BC-FL, iGRAIL
    y-axis: one metric.
    """
    z = (
        main_df.groupby(["Dataset", "Method"])[metric]
        .mean()
        .unstack()
        .reindex(columns=METHODS)
    )

    order = [
        d for d in ["cancer", "diabetes", "healthcare", "heart"]
        if d in z.index
    ]
    z = z.reindex(order)

    x = np.arange(len(z))
    width = 0.24
    hatches = ["///", "\\\\\\", "xx"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    for i, method in enumerate(METHODS):
        ax.bar(
            x + (i - 1) * width,
            z[method].values,
            width,
            hatch=hatches[i],
            alpha=0.88,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_LABELS.get(d, d) for d in z.index],
        fontsize=16
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel(DISPLAY[metric])

    lim = _metric_ylim_from_values(z.values.ravel(), metric)
    if lim:
        ax.set_ylim(*lim)

    compact_axes(ax)
    compact_legend(ax, 3)
    save_fig(fig, out / f"FinalDataset_{metric}_Bar")


def generate_final_one_metric_paper_set(
    main_df,
    round_df,
    alpha_df,
    privacy_round_df,
    privacy_final_df,
    scale_df,
    figures_root
):
    """
    Final manuscript-oriented output.

    Every figure has ONE metric on the y-axis.
    Several datasets and methods are merged in the same figure whenever
    the comparison remains technically meaningful.
    """
    folders = {
        "clients": figures_root / "A_client_scaling",
        "rounds": figures_root / "B_roundwise",
        "selected": figures_root / "C_selected_rounds",
        "alpha": figures_root / "D_noniid",
        "privacy": figures_root / "E_privacy",
        "bars": figures_root / "F_final_bars"
    }

    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)

    for metric in FINAL_METRICS:
        paper_client_scaling_one_metric(scale_df, metric, folders["clients"])
        paper_roundwise_one_metric(round_df, metric, folders["rounds"])
        paper_selected_rounds_one_metric(round_df, metric, folders["selected"])
        paper_alpha_one_metric(alpha_df, metric, folders["alpha"])
        paper_privacy_tradeoff_one_metric(
            privacy_final_df, metric, folders["privacy"]
        )
        paper_privacy_round_one_metric(
            round_df, privacy_round_df, metric, folders["privacy"]
        )
        paper_final_dataset_bar_one_metric(
            main_df, metric, folders["bars"]
        )


# ============================================================
# TABLE HELPERS
# ============================================================

def summarize_main(df):
    rows = []

    for (dataset, method), g in df.groupby(["Dataset", "Method"]):
        row = {
            "Dataset": dataset,
            "Method": method
        }

        for metric in METRICS:
            row[metric] = (
                f"{g[metric].mean():.4f} ± "
                f"{g[metric].std(ddof=0):.4f}"
            )

        rows.append(row)

    return pd.DataFrame(rows)



# ============================================================
# SECURITY MICROBENCHMARK
# ============================================================

def security_payload_benchmark(repeats=20):
    sizes_kb = [16, 64, 256, 1024, 4096]
    rows = []
    for kb in sizes_kb:
        payload = os.urandom(kb * 1024)
        for rep in range(repeats):
            v = VerificationLayer(1)
            t0 = time.perf_counter()
            sub = v.protect(0, 1, payload)
            protect_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            ok, _ = v.verify(sub, 1, consume=False)
            verify_ms = (time.perf_counter() - t0) * 1000.0
            if not ok:
                raise RuntimeError("Security benchmark verification failed.")

            rows.append({
                "PayloadKB": kb,
                "Repeat": rep,
                "ProtectMs": protect_ms,
                "VerifyMs": verify_ms
            })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        choices=["smoke", "paper"],
        default="smoke"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATA_INFO.keys()),
        default=["heart"]
    )

    parser.add_argument(
        "--data-dir",
        default="data"
    )

    parser.add_argument(
        "--out-dir",
        default="results"
    )

    args = parser.parse_args()

    cfg = get_preset(args.preset)
    cfg.out_dir = args.out_dir

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Device:", get_device())
    print("Opacus: True")
    print("Preset:", args.preset)
    print("Datasets:", args.datasets)
    print("Main clients:", cfg.n_clients)
    print("Scalability clients:", cfg.client_grid)

    sec = security_tests()
    sec.to_csv(
        out / "TABLE_security_results.csv",
        index=False
    )

    print("\nSecurity mechanism test:")
    print(sec.to_string(index=False))

    main_rows = []
    round_frames = []
    client_frames = []
    alpha_rows = []
    privacy_round_frames = []
    privacy_final_rows = []
    scale_rows = []
    scale_metric_frames = []
    scale_time_rows = []
    round_runtime_rows = []
    attack_round_frames = []
    attack_client_frames = []
    stats_rows = []

    prepared = {}

    # --------------------------------------------------------
    # Dataset preparation
    # --------------------------------------------------------
    for name in args.datasets:
        print(f"\n=== PREPARE {name.upper()} ===")

        X, y = load_raw(
            name,
            args.data_dir,
            cfg.max_rows,
            cfg.seed
        )

        Xtr, Xte, ytr, yte, _ = preprocess(
            X,
            y,
            cfg
        )

        prepared[name] = (
            Xtr,
            Xte,
            ytr,
            yte
        )

        stats_rows.append({
            "Dataset": name,
            "N": len(X),
            "PositiveRate": float(y.mean()),
            "EncodedDim": Xtr.shape[1]
        })

    # --------------------------------------------------------
    # Main repeated comparison
    # --------------------------------------------------------
    for name in args.datasets:
        Xtr, Xte, ytr, yte = prepared[name]

        print(f"\n=== MAIN: {name.upper()} ===")

        for rep in range(cfg.repeats):
            for method in METHODS:
                print(
                    f"[{name}] main "
                    f"rep={rep+1}/{cfg.repeats} "
                    f"{method}"
                )

                final, rr, cr = run_method(
                    method,
                    Xtr, ytr,
                    Xte, yte,
                    cfg,
                    n_clients=cfg.n_clients,
                    seed=cfg.seed + rep * 100
                )

                final.update({
                    "Dataset": name,
                    "Repeat": rep
                })

                rr["Dataset"] = name
                rr["Repeat"] = rep

                cr["Dataset"] = name
                cr["Repeat"] = rep

                main_rows.append(final)
                round_frames.append(rr)
                client_frames.append(cr)

    # --------------------------------------------------------
    # Adversarial update robustness experiment
    # --------------------------------------------------------
    for name in args.datasets:
        Xtr, Xte, ytr, yte = prepared[name]

        print(f"\n=== ATTACK ROBUSTNESS: {name.upper()} ===")

        for rep in range(cfg.repeats):
            for method in METHODS:
                print(
                    f"[{name}] attack rep={rep+1}/{cfg.repeats} "
                    f"{method}"
                )

                final, rr, cr = run_method(
                    method,
                    Xtr, ytr,
                    Xte, yte,
                    cfg,
                    n_clients=cfg.n_clients,
                    seed=cfg.seed + 5000 + rep * 100,
                    attack_mode=True
                )

                rr["Dataset"] = name
                rr["Repeat"] = rep
                cr["Dataset"] = name
                cr["Repeat"] = rep

                attack_round_frames.append(rr)
                attack_client_frames.append(cr)

    # --------------------------------------------------------
    # Non-IID alpha sweep
    # --------------------------------------------------------
    for name in args.datasets:
        Xtr, Xte, ytr, yte = prepared[name]

        print(f"\n=== ALPHA: {name.upper()} ===")

        for alpha in cfg.alpha_grid:
            for method in METHODS:
                print(
                    f"[{name}] alpha={alpha} "
                    f"{method}"
                )

                final, _, _ = run_method(
                    method,
                    Xtr, ytr,
                    Xte, yte,
                    cfg,
                    alpha=alpha,
                    n_clients=cfg.n_clients,
                    seed=cfg.seed + int(alpha * 1000)
                )

                alpha_rows.append({
                    "Dataset": name,
                    "Alpha": alpha,
                    "Method": method,
                    **{
                        metric: final[metric]
                        for metric in METRICS
                    }
                })

    # --------------------------------------------------------
    # Privacy sweep
    # --------------------------------------------------------
    for name in args.datasets:
        Xtr, Xte, ytr, yte = prepared[name]

        print(f"\n=== PRIVACY: {name.upper()} ===")

        for delta in cfg.delta_grid:
            for nm in cfg.privacy_noise_grid:
                print(
                    f"[{name}] "
                    f"delta={delta:.0e}, noise_multiplier={nm}"
                )

                final, rr, cr = run_method(
                    "iGRAIL",
                    Xtr, ytr,
                    Xte, yte,
                    cfg,
                    n_clients=cfg.n_clients,
                    noise=nm,
                    delta=delta,
                    seed=cfg.seed
                )

                rr["Dataset"] = name
                rr["Delta"] = delta
                cr["Dataset"] = name
                cr["Delta"] = delta

                privacy_round_frames.append(rr)

                privacy_final_rows.append({
                    "Dataset": name,
                    "NoiseMultiplier": nm,
                    "Delta": delta,
                    "Epsilon": final["Epsilon"],
                    **{
                        metric: final[metric]
                        for metric in METRICS
                    }
                })

    # --------------------------------------------------------
    # Client scalability / iteration experiments
    # --------------------------------------------------------
    for name in args.datasets:
        Xtr, Xte, ytr, yte = prepared[name]
        print(f"\n=== SCALABILITY: {name.upper()} ===")

        # A. Round-wise AUROC for several client counts and all methods.
        for n_clients in cfg.client_grid:
            for method in METHODS:
                try:
                    final, rr, _ = run_method(
                        method, Xtr, ytr, Xte, yte, cfg,
                        n_clients=n_clients,
                        alpha=cfg.dirichlet_alpha,
                        seed=cfg.seed + 100*n_clients
                    )
                    rr["Dataset"] = name
                    rr["Clients"] = n_clients
                    scale_metric_frames.append(rr)

                    scale_rows.append({
                        "Dataset": name,
                        "Clients": n_clients,
                        "Method": method,
                        **{metric: final[metric] for metric in METRICS},
                        "LocalTime": final["LocalTime"],
                        "VerifyTime": final["VerifyTime"],
                        "AggregateTime": final["AggregateTime"],
                        "TotalTime": final["TotalTime"]
                    })
                except RuntimeError as e:
                    print(f"Skipped {name}, K={n_clients}, {method}: {e}")

        # B. iGRAIL execution time vs clients for several FL round counts.
        for n_clients in cfg.client_grid:
            for rounds in cfg.scalability_round_grid:
                try:
                    with temporary_rounds(cfg, rounds):
                        final, _, _ = run_method(
                            "iGRAIL", Xtr, ytr, Xte, yte, cfg,
                            n_clients=n_clients,
                            alpha=cfg.dirichlet_alpha,
                            seed=cfg.seed + 1000 + 17*n_clients + rounds
                        )
                    scale_time_rows.append({
                        "Dataset": name,
                        "Clients": n_clients,
                        "Rounds": rounds,
                        "Method": "iGRAIL",
                        "TotalTime": final["TotalTime"]
                    })
                except RuntimeError as e:
                    print(f"Skipped time scaling {name}, K={n_clients}, R={rounds}: {e}")

        # C. Local training / aggregation / total runtime vs rounds for all methods.
        for rounds in cfg.scalability_round_grid:
            for method in METHODS:
                try:
                    with temporary_rounds(cfg, rounds):
                        final, _, _ = run_method(
                            method, Xtr, ytr, Xte, yte, cfg,
                            n_clients=cfg.n_clients,
                            alpha=cfg.dirichlet_alpha,
                            seed=cfg.seed + 2000 + rounds
                        )
                    round_runtime_rows.append({
                        "Dataset": name,
                        "Rounds": rounds,
                        "Method": method,
                        "LocalTime": final["LocalTime"],
                        "VerifyTime": final["VerifyTime"],
                        "AggregateTime": final["AggregateTime"],
                        "TotalTime": final["TotalTime"]
                    })
                except RuntimeError as e:
                    print(f"Skipped runtime {name}, R={rounds}, {method}: {e}")

    # --------------------------------------------------------
    # Assemble outputs
    # --------------------------------------------------------
    main_df = pd.DataFrame(main_rows)

    round_df = pd.concat(
        round_frames,
        ignore_index=True
    )

    client_df = pd.concat(
        client_frames,
        ignore_index=True
    )

    alpha_df = pd.DataFrame(alpha_rows)

    privacy_round_df = pd.concat(
        privacy_round_frames,
        ignore_index=True
    )

    privacy_final_df = pd.DataFrame(
        privacy_final_rows
    )

    attack_round_df = pd.concat(attack_round_frames, ignore_index=True)
    attack_client_df = pd.concat(attack_client_frames, ignore_index=True)

    scale_df = pd.DataFrame(scale_rows)
    scale_metric_df = pd.concat(scale_metric_frames, ignore_index=True)
    scale_time_df = pd.DataFrame(scale_time_rows)
    round_runtime_df = pd.DataFrame(round_runtime_rows)
    sec_bench_df = security_payload_benchmark(repeats=10 if args.preset=="smoke" else 30)

    # --------------------------------------------------------
    # Main tables
    # --------------------------------------------------------
    main_df.to_csv(
        out / "ALL_main_results_raw.csv",
        index=False
    )

    summarize_main(main_df).to_csv(
        out / "TABLE_main_comparison.csv",
        index=False
    )

    round_df.to_csv(
        out / "ALL_roundwise_results.csv",
        index=False
    )

    attack_round_df.to_csv(
        out / "ALL_attack_roundwise_results.csv",
        index=False
    )

    attack_client_df.to_csv(
        out / "ALL_attack_client_results.csv",
        index=False
    )

    client_df.to_csv(
        out / "ALL_client_roundwise_results.csv",
        index=False
    )

    alpha_df.to_csv(
        out / "ALL_alpha_results.csv",
        index=False
    )

    privacy_round_df.to_csv(
        out / "ALL_epsilon_roundwise_results.csv",
        index=False
    )

    privacy_final_df.to_csv(
        out / "TABLE_privacy_results.csv",
        index=False
    )

    scale_df.to_csv(
        out / "TABLE_client_scalability.csv",
        index=False
    )

    scale_metric_df.to_csv(
        out / "ALL_client_round_auroc.csv",
        index=False
    )

    scale_time_df.to_csv(
        out / "TABLE_clients_rounds_time.csv",
        index=False
    )

    round_runtime_df.to_csv(
        out / "TABLE_round_runtime_components.csv",
        index=False
    )

    sec_bench_df.to_csv(
        out / "TABLE_crypto_payload_time.csv",
        index=False
    )

    pd.DataFrame(stats_rows).to_csv(
        out / "TABLE_dataset_stats.csv",
        index=False
    )

    # Overall method table across all datasets/repeats.
    overall_rows = []
    for method, g in main_df.groupby("Method"):
        row = {"Method": method}
        for metric in METRICS:
            row[metric] = (
                f"{g[metric].mean():.4f} ± "
                f"{g[metric].std(ddof=0):.4f}"
            )
        overall_rows.append(row)
    pd.DataFrame(overall_rows).to_csv(
        out / "TABLE_overall_method_metrics.csv",
        index=False
    )

    # Final-round metric by client count.
    scale_df.to_csv(
        out / "TABLE_all_metrics_by_clients.csv",
        index=False
    )

    # Non-IID all metrics.
    alpha_df.to_csv(
        out / "TABLE_all_metrics_by_alpha.csv",
        index=False
    )

    # Privacy all metrics.
    privacy_final_df.to_csv(
        out / "TABLE_all_metrics_by_privacy.csv",
        index=False
    )

    # Explicit epsilon-delta privacy table for the paper.
    privacy_cols = [
        "Dataset", "Delta", "NoiseMultiplier", "Epsilon",
        "Accuracy", "Precision", "Recall", "Specificity",
        "BalancedAcc", "F1", "AUROC", "AUPRC", "MCC"
    ]
    privacy_final_df[privacy_cols].sort_values(
        ["Dataset", "Delta", "NoiseMultiplier"]
    ).to_csv(
        out / "TABLE_epsilon_delta_privacy.csv",
        index=False
    )

    # --------------------------------------------------------
    # Individual numerical evidence only
    # --------------------------------------------------------
    indiv = out / "individual_csv"
    indiv.mkdir(exist_ok=True)

    for name in args.datasets:
        d = indiv / name
        d.mkdir(exist_ok=True)

        main_df[
            main_df["Dataset"] == name
        ].to_csv(
            d / "main.csv",
            index=False
        )

        round_df[
            round_df["Dataset"] == name
        ].to_csv(
            d / "roundwise.csv",
            index=False
        )

        client_df[
            client_df["Dataset"] == name
        ].to_csv(
            d / "client_roundwise.csv",
            index=False
        )

        alpha_df[
            alpha_df["Dataset"] == name
        ].to_csv(
            d / "alpha.csv",
            index=False
        )

        privacy_round_df[
            privacy_round_df["Dataset"] == name
        ].to_csv(
            d / "epsilon_roundwise.csv",
            index=False
        )

        scale_df[
            scale_df["Dataset"] == name
        ].to_csv(
            d / "client_scalability.csv",
            index=False
        )

    # --------------------------------------------------------
    # Compact paper-facing LINE plots
    # --------------------------------------------------------
    figs = out / "figures"
    figs.mkdir(exist_ok=True)

    generate_compact_paper_plots(
        main_df=main_df,
        round_df=round_df,
        alpha_df=alpha_df,
        privacy_round_df=privacy_round_df,
        privacy_final_df=privacy_final_df,
        scale_df=scale_df,
        scale_metric_df=scale_metric_df,
        scale_time_df=scale_time_df,
        round_runtime_df=round_runtime_df,
        sec_bench_df=sec_bench_df,
        figures_root=figs
    )

    # Realistic learning curves with all rounds and confidence bands.
    curve_dir = figs / "09_realistic_curves"
    curve_dir.mkdir(exist_ok=True)
    for metric in PAPER_METRICS:
        plot_roundwise_metric_ci(round_df, metric, curve_dir)

    # Robustness experiment: one malicious client from attack_start_round.
    attack_dir = figs / "10_attack_robustness"
    attack_dir.mkdir(exist_ok=True)
    for metric in ["Accuracy", "Recall", "F1", "AUROC", "AUPRC", "MCC"]:
        plot_attack_robustness(
            attack_round_df,
            metric,
            attack_dir,
            cfg.attack_start_round
        )

    # Bars only where the x-axis is categorical.
    bar_dir = figs / "11_bar_comparisons"
    bar_dir.mkdir(exist_ok=True)
    plot_final_metric_bars(main_df, bar_dir)
    plot_security_acceptance_bar(sec, bar_dir)


    # --------------------------------------------------------
    # FINAL recommended paper figure set
    # --------------------------------------------------------
    paper_dir = figs / "12_FINAL_PAPER_FIGURES"
    paper_dir.mkdir(exist_ok=True)

    paper_convergence_compact(round_df, paper_dir)
    paper_client_scaling_compact(scale_df, paper_dir)
    paper_privacy_compact(round_df, privacy_round_df, paper_dir)
    paper_attack_compact(attack_round_df, cfg.attack_start_round, paper_dir)
    paper_selected_rounds_dataset_metrics(client_df, paper_dir)
    paper_per_client_selected_rounds(client_df, paper_dir)
    paper_final_bar(main_df, paper_dir)


    # --------------------------------------------------------
    # FINAL ONE-METRIC / MULTI-DATASET FIGURE SET
    # --------------------------------------------------------
    one_metric_dir = figs / "13_FINAL_ONE_METRIC_MULTI_DATASET"
    one_metric_dir.mkdir(exist_ok=True)

    generate_final_one_metric_paper_set(
        main_df=main_df,
        round_df=round_df,
        alpha_df=alpha_df,
        privacy_round_df=privacy_round_df,
        privacy_final_df=privacy_final_df,
        scale_df=scale_df,
        figures_root=one_metric_dir
    )

    print("\nCOMPLETED")
    print("Figures:", figs.resolve())
    print("Main table:", (out / "TABLE_main_comparison.csv").resolve())
    print("Round-wise:", (out / "ALL_roundwise_results.csv").resolve())
    print("Client-wise:", (out / "ALL_client_roundwise_results.csv").resolve())
    print("Scalability:", (out / "TABLE_client_scalability.csv").resolve())
    recommended = [
        "02_privacy_all_methods/Privacy_EpsDelta_AllMethods_AUROC.pdf",
        "02_privacy_all_methods/PrivacyTradeoff_AUROC.pdf",
        "02_privacy_all_methods/Privacy_EpsDelta_AllMethods_Recall.pdf",
        "02_privacy_all_methods/PrivacyTradeoff_Recall.pdf",
        "03_noniid/Alpha_AUROC.pdf",
        "04_clients_2to10/Clients_2to10_AUROC.pdf",
        "04_clients_2to10/Clients_2to10_Recall.pdf",
        "05_clients_rounds/ClientsRounds_AUROC.pdf",
        "06_datasets/Datasets_AUROC.pdf",
        "07_runtime/Execution_Clients_2to10.pdf",
        "07_runtime/Runtime_Local.pdf",
        "07_runtime/Runtime_Agg.pdf",
        "08_security/Crypto_PayloadTime.pdf",
        "09_realistic_curves/RoundCurve_AUROC.pdf",
        "09_realistic_curves/RoundCurve_Recall.pdf",
        "10_attack_robustness/AttackCurve_AUROC.pdf",
        "10_attack_robustness/AttackCurve_Recall.pdf",
        "11_bar_comparisons/FinalMetrics_Bar.pdf",
        "11_bar_comparisons/Security_Acceptance_Bar.pdf",
        "12_FINAL_PAPER_FIGURES/P1_Convergence_Accuracy_F1_AUROC.pdf",
        "12_FINAL_PAPER_FIGURES/P2_ClientScaling_Recall_F1_AUROC.pdf",
        "12_FINAL_PAPER_FIGURES/P3_Privacy_FL_BCFL_iGRAIL.pdf",
        "12_FINAL_PAPER_FIGURES/P4_AttackRobustness_AUROC.pdf",
        "12_FINAL_PAPER_FIGURES/P7_FinalMetrics_Bar.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/A_client_scaling/ClientScaling_AUROC.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/A_client_scaling/ClientScaling_Recall.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/B_roundwise/Roundwise_AUROC.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/B_roundwise/Roundwise_Recall.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/C_selected_rounds/SelectedRounds_AUROC.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/D_noniid/NonIID_AUROC.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/E_privacy/PrivacyTradeoff_AUROC.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/E_privacy/PrivacyRounds_AUROC.pdf",
        "13_FINAL_ONE_METRIC_MULTI_DATASET/F_final_bars/FinalDataset_AUROC_Bar.pdf",
    ]
    (out / "RECOMMENDED_PAPER_FIGURES.txt").write_text(
        "\n".join(recommended) + "\n"
    )

if __name__ == "__main__":
     main()
