"""Gnom-0.1 — tiny damage net (battlefield zoo): 440 neurons, pure numpy.

    12 features -> 200 -> 160 -> 64 -> 15 -> 1  (200+160+64+15+1 = 440)

Predicts measured PPL damage (group_damage_sweep.py labels) from cheap
per-tensor features. ReLU hidden, linear out, full-batch Adam, MSE + L2.
Trains in seconds on CPU.

Features (12): type one-hot (6: attn_gate, attn_qkv, attn_output, ffn_gate,
ffn_up, ffn_down) + layer_pos + log_n_elements + timp_mean + timp_max +
ssim_fragility + energy_concentration.
"""
import numpy as np

DIMS = [12, 200, 160, 64, 15, 1]


def init_params(dims, seed=7):
    rng = np.random.default_rng(seed)
    ps = []
    for a, b in zip(dims[:-1], dims[1:]):
        ps.append({"W": rng.normal(0, np.sqrt(2 / a), (a, b)),
                   "b": np.zeros(b)})
    return ps


def _mish(x):
    return x * np.tanh(np.log1p(np.exp(np.clip(x, -30, 30))))


def _mish_d(x):
    s = 1 / (1 + np.exp(-np.clip(x, -30, 30)))
    t = np.tanh(np.log1p(np.exp(np.clip(x, -30, 30))))
    return t + x * s * (1 - t * t)


def forward(ps, X, activation="relu", train_drop=0.0, rng=None):
    acts, pre = [X], []
    for i, p in enumerate(ps):
        z = acts[-1] @ p["W"] + p["b"]
        pre.append(z)
        if i == len(ps) - 1:
            acts.append(z)
        else:
            h = _mish(z) if activation == "mish" else np.maximum(z, 0)
            if train_drop > 0 and rng is not None:
                h = h * (rng.random(h.shape) >= train_drop) / (1 - train_drop)
            acts.append(h)
    return acts, pre


def backward_delta(delta, z, activation):
    if activation == "mish":
        return delta * _mish_d(z)
    return delta * (z > 0)


def train(ps, X, y, lr=3e-3, epochs=2000, l2=1e-5, log_every=500,
          noise_std=0.0, sample_w=None, seed=99, activation="mish",
          dropout=0.0, val_frac=0.2, patience=None):
    """Full-batch Adam + RealMLP-inspired defaults for small tabular data.

    - Mish activation (best for regression per RealMLP ablations)
    - dropout on hidden layers, targets standardized outside
    - best-epoch revert on a validation split (no fixed-epoch gambling);
      patience=None trains full schedule and reverts to best val.
    - noise_std: feature jitter (anti-parrot); sample_w: label trust.
    """
    rng = np.random.default_rng(seed)
    if sample_w is None:
        sample_w = np.ones(len(X))
    sample_w = np.asarray(sample_w, dtype=np.float64)
    n = len(X)
    nval = max(4, int(n * val_frac))
    vidx = rng.choice(n, nval, replace=False)
    mask = np.ones(n, bool)
    mask[vidx] = False
    Xtr, ytr, wtr = X[mask], y[mask], sample_w[mask]
    Xva, yva = X[~mask], y[~mask]
    m = [{k: np.zeros_like(v) for k, v in p.items()} for p in ps]
    v = [{k: np.zeros_like(v) for k, v in p.items()} for p in ps]
    best = (float("inf"), None, 0)
    bad = 0
    for ep in range(1, epochs + 1):
        Xn = Xtr + rng.normal(0, noise_std, Xtr.shape) if noise_std > 0 else Xtr
        acts, pre = forward(ps, Xn, activation, dropout, rng)
        err = (acts[-1].ravel() - ytr) * wtr / wtr.sum()
        delta = err[:, None]
        for i in reversed(range(len(ps))):
            gW = acts[i].T @ delta + l2 * ps[i]["W"]
            gb = delta.sum(axis=0)
            if i > 0:
                delta = backward_delta(delta @ ps[i]["W"].T, pre[i - 1],
                                       activation)
            for k, g in (("W", gW), ("b", gb)):
                m[i][k] = 0.9 * m[i][k] + 0.1 * g
                v[i][k] = 0.95 * v[i][k] + 0.05 * g * g  # beta2=0.95 (fastai)
                mh = m[i][k] / (1 - 0.9 ** ep)
                vh = v[i][k] / (1 - 0.95 ** ep)
                ps[i][k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        pv = forward(ps, Xva, activation)[0][-1].ravel()
        vl = float(np.mean((pv - yva) ** 2))
        if vl < best[0]:
            import copy
            best = (vl, copy.deepcopy(ps), ep)
            bad = 0
        else:
            bad += 1
        if patience and bad >= patience:
            break
        if ep % log_every == 0 or ep == 1:
            print(f"  ep {ep}: val MSE={vl:.5f} (best {best[0]:.5f} @ {best[2]})",
                  flush=True)
    if best[1] is not None:
        for i in range(len(ps)):
            ps[i] = best[1][i]
    return ps


def standardize(X):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    return (X - mu) / sd, mu, sd


# ------------------------------------------------- Gnom-MIXED (mixed-effects)
# Shared trunk learns general physics on ALL labels; per-group bias absorbs
# individuality (James-Stein shrinkage: b = mean_resid * n/(n+lam) — with
# n=1-2 labels per group only screaming evidence survives). Fitted by
# backfitting loop (the principled 'looped' idea): trunk <-> biases.
# Tier features (base/drop bpw) let one model learn the damage CURVE.

def fit_mixed(X, y, gidx, n_groups, dims, lam=1.0, iters=3, seed=7, **kw):
    kw = dict(kw)
    kw.setdefault("activation", "mish")
    kw.setdefault("dropout", 0.0)
    ymu, ysd = float(y.mean()), float(y.std() + 1e-12)
    ys = (y - ymu) / ysd
    Xs, xmu, xsd = standardize(X)
    bias = np.zeros(n_groups)
    trunk = None
    for it in range(iters):
        resid = ys - bias[gidx]
        mu_r, sd_r = float(resid.mean()), float(resid.std() + 1e-12)
        ps = init_params(dims, seed=seed + it)
        w = np.abs((resid - mu_r) / sd_r) + 0.05
        train(ps, Xs, (resid - mu_r) / sd_r, sample_w=w, seed=seed + it,
              **kw)
        trunk = (ps, mu_r, sd_r)
        pred = forward(ps, Xs, kw.get("activation", "mish"))[0][-1].ravel() \
            * sd_r + mu_r
        r2 = ys - pred
        new_bias = np.zeros(n_groups)
        for g in range(n_groups):
            m = gidx == g
            if m.any():
                new_bias[g] = r2[m].mean() * m.sum() / (m.sum() + lam)
        shift = float(np.abs(new_bias - bias).max())
        bias = new_bias
        print(f"  backfit {it + 1}: max bias shift={shift:.5f}, "
              f"bias std={float(bias.std()):.4f}", flush=True)
        if shift < 1e-4:
            break
    return {"trunk": trunk[0], "tmu": trunk[1], "tsd": trunk[2],
            "bias": bias, "ymu": ymu, "ysd": ysd,
            "xmu": xmu, "xsd": xsd}


def mixed_predict(mp, X, gidx, n_groups, activation="mish"):
    ps, mu_r, sd_r = mp["trunk"], mp["tmu"], mp["tsd"]
    Xs = (X - mp["xmu"]) / mp["xsd"]
    base = forward(ps, Xs, activation)[0][-1].ravel() * sd_r + mu_r
    b = np.zeros(len(X))
    known = gidx < n_groups
    b[known] = mp["bias"][gidx[known]]
    return (base + b) * mp["ysd"] + mp["ymu"]


def boost(X, y, dims, rounds=3, shrinkage=0.3, seed=7,
          val_frac=0.25, deep_alpha=0.2, **kw):
    """Looped Gnom: iterative residual refinement (the honest transfer of the
    looped-transformer idea to tabular data — same tiny block re-applied,
    each round fitting what previous rounds left unexplained).

    2026 lessons baked in:
    - validation-gated rounds (anti-'overthinking': stop adding rounds when
      heldout worsens — more loops can degrade, not just saturate);
    - deep supervision (LOTUS-style): each round's target blends the residual
      with the original target, so intermediate rounds stay grounded.
    Returns (ensemble_params_list, ymu, ysd, rounds_used)."""
    rng = np.random.default_rng(seed)
    ymu, ysd = float(y.mean()), float(y.std() + 1e-12)
    ys = (y - ymu) / ysd
    n = len(X)
    nval = max(4, int(n * val_frac))
    vidx = rng.choice(n, nval, replace=False)
    mask = np.ones(n, bool)
    mask[vidx] = False
    resid, ensemble = ys.copy(), []
    best_val, best_ens = float("inf"), []
    for r in range(rounds):
        ps = init_params(dims, seed=seed + r)
        # deep supervision: residual blended toward the original target
        tgt = (1 - deep_alpha) * resid + deep_alpha * ys
        w = np.abs(tgt[mask]) + 0.05 * np.abs(tgt[mask]).max()
        train(ps, X[mask], tgt[mask], sample_w=w, seed=seed + r, **kw)
        pred = forward(ps, X, kw.get("activation", "mish"))[0][-1].ravel()
        resid = resid - shrinkage * pred
        ensemble.append(ps)
        cur = sum(shrinkage * forward(p, X[~mask],
                                     kw.get("activation", "mish"))[0][-1].ravel()
                  for p in ensemble)
        vl = float(np.mean((cur - ys[~mask]) ** 2))
        print(f"  round {r + 1}: heldout MSE={vl:.5f}", flush=True)
        if vl < best_val:
            best_val, best_ens = vl, list(ensemble)
        else:
            print(f"  overthinking gate: round {r + 1} worsens "
                  f"({vl:.5f} > {best_val:.5f}) — stop", flush=True)
            break
    return best_ens, ymu, ysd, len(best_ens)


def boost_predict(ensemble, X, ymu, ysd, shrinkage=0.3, activation="mish"):
    out = np.zeros(len(X))
    for ps in ensemble:
        out += shrinkage * forward(ps, X, activation)[0][-1].ravel()
    return out * ysd + ymu


# ---------------------------------------------------------------- Gnom-MoE
# One tiny expert per tensor type (9 feats: type one-hot dropped, the router
# is HARD — tensor type itself. No learned gate, no routing collapse, and
# each expert learns its own type's damage physics with ~45 params).
# Total: 6 experts x 45 = 270 params — same budget as the single net.

MOE_DIMS = [9, 4, 1]
MOE_FEAT_IDX = [6, 7, 8, 9, 10, 11, 12, 13, 14]  # layer..w-stats (no one-hot)


def moe_split(X):
    """Hard router: group row indices by argmax of type one-hot (cols 0-5)."""
    owners = np.argmax(X[:, :6], axis=1)
    return {t: np.where(owners == t)[0] for t in range(6) if (owners == t).any()}


def moe_train(X, y, seed=7, **kw):
    kw = dict(kw)
    kw.setdefault("activation", "mish")
    kw.setdefault("dropout", 0.0)
    experts = {}
    for t, idx in moe_split(X).items():
        Xe = standardize(X[idx][:, MOE_FEAT_IDX])[0]
        ye = y[idx]
        mu, sd = float(ye.mean()), float(ye.std() + 1e-12)
        ps = init_params(MOE_DIMS, seed=seed + t)
        w = np.abs((ye - mu) / sd) + 0.05
        train(ps, Xe, (ye - mu) / sd, sample_w=w, seed=seed + t, **kw)
        experts[t] = {"ps": ps, "mu": mu, "sd": sd,
                      "xmu": Xe.mean(axis=0) * 0, "xsd": np.ones(9)}
    # store standardizer per expert for predict
    for t, idx in moe_split(X).items():
        Xm = X[idx][:, MOE_FEAT_IDX]
        experts[t]["xmu"] = Xm.mean(axis=0)
        experts[t]["xsd"] = Xm.std(axis=0) + 1e-12
    return experts


def moe_predict(experts, X, activation="mish", fallback=None):
    out = np.zeros(len(X))
    owners = np.argmax(X[:, :6], axis=1)
    for t, e in experts.items():
        idx = np.where(owners == t)[0]
        if len(idx) == 0:
            continue
        Xe = (X[idx][:, MOE_FEAT_IDX] - e["xmu"]) / e["xsd"]
        out[idx] = forward(e["ps"], Xe, activation)[0][-1].ravel() * e["sd"] \
            + e["mu"]
    return out


# ------------------------------------------------------- Gnom-LIQUID
# ------------------------------------------------- Gnom-MoE-SHARED
# Flash-Next mirror: 1 shared expert (type-independent signal: position,
# size, weight stats) + 6 routed experts (type physics). Output = shared +
# routed, exactly like Qwen's shared+routed sum. The shared expert also
# rescues types unseen in training (falls back to shared alone).

def moe_train_shared(X, y, seed=7, **kw):
    kw = dict(kw)
    kw.setdefault("activation", "mish")
    kw.setdefault("dropout", 0.0)
    experts = moe_train(X, y, seed=seed, **kw)
    Xs, _, _ = standardize(X[:, MOE_FEAT_IDX])
    mu, sd = float(y.mean()), float(y.std() + 1e-12)
    ps = init_params(MOE_DIMS, seed=seed + 99)
    w = np.abs((y - mu) / sd) + 0.05
    train(ps, Xs, (y - mu) / sd, sample_w=w, seed=seed + 99, **kw)
    Xm = X[:, MOE_FEAT_IDX]
    experts["shared"] = {"ps": ps, "mu": mu, "sd": sd,
                         "xmu": Xm.mean(axis=0), "xsd": Xm.std(axis=0) + 1e-12}
    return experts


def moe_predict_shared(experts, X, activation="mish", w_shared=0.5):
    out = (1 - w_shared) * moe_predict(
        {t: e for t, e in experts.items() if t != "shared"}, X, activation)
    e = experts["shared"]
    Xe = (X[:, MOE_FEAT_IDX] - e["xmu"]) / e["xsd"]
    out += w_shared * (forward(e["ps"], Xe, activation)[0][-1].ravel()
                       * e["sd"] + e["mu"])
    return out


# ------------------------------------------------- Gnom-QSA (feature gate)
# QSA analog for tabular data: a lightweight indexer scores the 15 feature
# "micro-blocks" per sample and keeps the signal; top-K masking is replaced
# by a learned sigmoid gate (differentiable top-K). Trained second-stage on
# frozen single-net residuals — like the indexer serving the attention.

def qsa_gate_train(X, resid, seed=7, lr=0.05, epochs=500, l2=1e-4,
                   log_every=250):
    rng = np.random.default_rng(seed)
    Xs, _, _ = standardize(X)
    Gf = rng.normal(0, 0.1, (15,))
    bf = 0.0
    mG, mb = np.zeros(15), 0.0
    n = len(X)
    for ep in range(1, epochs + 1):
        g = 1 / (1 + np.exp(-(Xs * Gf).sum(axis=1) - bf))
        # target: normalized residual energy — the gate learns which samples
        # deserve more compute (adaptive-compute à la RecurTrace halting head)
        tgt = (resid ** 2) / ((resid ** 2).max() + 1e-12)
        loss_grad = (g - tgt) / n
        gg = (Xs * loss_grad[:, None]).mean(axis=0) + l2 * Gf
        gb = float(loss_grad.mean())
        mG = 0.9 * mG + 0.1 * gg
        mb = 0.9 * mb + 0.1 * gb
        Gf -= lr * mG
        bf -= lr * mb
        if ep % log_every == 0 or ep == 1:
            print(f"  qsa ep {ep}: gate mean={float(g.mean()):.3f}",
                  flush=True)
    return {"Gf": Gf, "bf": float(bf)}


def qsa_apply(gate, X):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    g = 1 / (1 + np.exp(-(((X - mu) / sd) * gate["Gf"]).sum(axis=1)
                         - gate["bf"]))
    return g  # per-sample salience in [0,1]
# Boosted MoE: every boost round is a 6-expert MoE fitting the residual.
# Single + Loop + MoE in one beast. Rounds gated by heldout (anti-overthink).

def liquid_train_gate(X, y, experts, lr=0.05, epochs=800, l2=1e-4, seed=7,
                      log_every=400):
    rng = np.random.default_rng(seed)
    E = np.stack([moe_predict({t: e}, X) for t, e in
                  sorted(experts.items())], axis=1)  # n x 6 (NaN→0 below)
    E = np.nan_to_num(E)
    mu, sd = E.mean(axis=0), E.std(axis=0) + 1e-12
    En = (E - mu) / sd
    Xs, _, _ = standardize(X)
    G = rng.normal(0, 0.1, (15, 6))
    b = np.zeros(6)
    mG, mb = np.zeros_like(G), np.zeros_like(b)
    n = len(X)
    for ep in range(1, epochs + 1):
        L = Xs @ G + b
        L -= L.max(axis=1, keepdims=True)
        W = np.exp(L)
        W /= W.sum(axis=1, keepdims=True)
        pred = (W * En).sum(axis=1) * sd.mean() + mu.mean()
        err = (pred - y) / n
        dW = (err[:, None] * En) * sd.mean()
        dL = W * (dW - (dW * W).sum(axis=1, keepdims=True))
        gG = Xs.T @ dL + l2 * G
        gb = dL.sum(axis=0)
        for M, g, P in ((mG, gG, G), (mb, gb, b)):
            M[:] = 0.9 * M + 0.1 * g
            P[:] -= lr * M
        if ep % log_every == 0 or ep == 1:
            print(f"  gate ep {ep}: RMSE={float(np.sqrt(np.mean((pred - y) ** 2))):.5f}",
                  flush=True)
    return {"G": G, "b": b, "emu": mu, "esd": sd}


def liquid_predict(experts, gate, X):
    E = np.stack([moe_predict({t: e}, X) for t, e in
                  sorted(experts.items())], axis=1)
    E = np.nan_to_num(E)
    En = (E - gate["emu"]) / gate["esd"]
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    L = ((X - mu) / sd) @ gate["G"] + gate["b"]
    L -= L.max(axis=1, keepdims=True)
    W = np.exp(L)
    W /= W.sum(axis=1, keepdims=True)
    return ((W * En).sum(axis=1) * gate["esd"].mean() + gate["emu"].mean())
# Boosted MoE: every boost round is a 6-expert MoE fitting the residual.
# Single + Loop + MoE in one beast. Rounds gated by heldout (anti-overthink).

# ------------------------------------------------------- Gnom-ULTIMATE
# Boosted MoE: every boost round is a 6-expert MoE fitting the residual.
def ultimate_train(X, y, rounds=3, shrinkage=0.3, seed=7, val_frac=0.25,
                   **kw):
    rng = np.random.default_rng(seed)
    n = len(X)
    nval = max(4, int(n * val_frac))
    vidx = rng.choice(n, nval, replace=False)
    mask = np.ones(n, bool)
    mask[vidx] = False
    resid = y.copy()
    ensemble, best_val, best_ens = [], float("inf"), []
    for r in range(rounds):
        experts = moe_train(X[mask], resid[mask], seed=seed + r, **kw)
        pred = moe_predict(experts, X)
        resid = resid - shrinkage * pred
        ensemble.append(experts)
        cur = np.zeros(n)
        for e in ensemble:
            cur += shrinkage * moe_predict(e, X)
        vl = float(np.mean((cur[~mask] - y[~mask]) ** 2))
        print(f"  ULT round {r + 1}: heldout MSE={vl:.5f}", flush=True)
        if vl < best_val:
            best_val, best_ens = vl, list(ensemble)
        else:
            print(f"  gate: round {r + 1} worsens — stop", flush=True)
            break
    return best_ens, len(best_ens)


def ultimate_predict(ensemble, X, shrinkage=0.3):
    out = np.zeros(len(X))
    for e in ensemble:
        out += shrinkage * moe_predict(e, X)
    return out


TYPES = ["attn_gate", "attn_qkv", "attn_output", "ffn_gate", "ffn_up",
         "ffn_down"]


def featurize(name, n_el, timp_mean, timp_max, frag=0.0, econc=0.0,
              n_layers=28, w=None):
    """Two books: imatrix features + model weight statistics.
    w: raw weight array (optional) — std, kurtosis, max/mean outlierness.
    All scale-dependent features are meant to be z-scored per model later."""
    parts = name.split(".")
    ttype = parts[2] if parts[0] == "blk" and len(parts) > 2 else "other"
    layer = int(parts[1]) / max(1, n_layers) if parts[0] == "blk" else 0.0
    onehot = [1.0 if ttype == t else 0.0 for t in TYPES]
    feats = onehot + [layer, float(np.log10(n_el + 1)),
                      float(timp_mean), float(timp_max),
                      float(frag), float(econc)]
    if w is not None:
        f = np.asarray(w, dtype=np.float64).ravel()
        std = float(f.std())
        m4 = float(((f - f.mean()) ** 4).mean())
        kurt = m4 / (std ** 4 + 1e-12) - 3.0
        mxm = float(np.abs(f).max() / (np.abs(f).mean() + 1e-12))
        feats += [np.log10(std + 1e-12), float(np.clip(kurt, -2, 50)) / 10.0,
                  np.log10(mxm + 1)]
    else:
        feats += [0.0, 0.0, 0.0]
    return feats


N_FEATS = 15
