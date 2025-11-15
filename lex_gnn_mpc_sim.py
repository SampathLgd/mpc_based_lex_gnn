# lex_gnn_mpc_sim.py
# Simulated ASTRA-like MPC prototype for one LEX-GNN layer (inference).
# Updated to remove NumPy deprecation warnings and add simple CLI args.
# Requires: numpy, pandas

import argparse
import numpy as np
import pandas as pd

# --- config defaults ---
SCALE = 2**16   # fractional bits
FRAC_BITS = 16

# --- fixed-point helpers ---
def to_fixed(x):
    """
    Convert float numpy array to fixed-point int64 bit-patterns then view as uint64.
    """
    xf = np.asarray(x, dtype=np.float64)
    xi = np.round(xf * SCALE).astype(np.int64)
    return xi.view(np.uint64)

def from_fixed(x_int):
    """
    Convert uint64 array (two's complement representation) back to float.
    """
    xi = x_int.view(np.int64).astype(np.float64)
    return xi / SCALE

def ring_add(a, b):
    return (a.astype(np.uint64) + b.astype(np.uint64)).astype(np.uint64)

def ring_sub(a, b):
    return (a.astype(np.uint64) - b.astype(np.uint64)).astype(np.uint64)

# --- additive 3-share simulation ---
def share_secret(x):
    """
    x is expected to be a uint64 numpy array (fixed-point).
    Returns three uint64 shares.
    """
    s0 = np.random.randint(0, 2**63, size=x.shape, dtype=np.uint64)
    s1 = np.random.randint(0, 2**63, size=x.shape, dtype=np.uint64)
    s2 = ring_sub(ring_sub(x, s0), s1)
    return [s0, s1, s2]

def reconstruct(shares):
    ssum = ring_add(shares[0], shares[1])
    ssum = ring_add(ssum, shares[2])
    return ssum

def share_from_float(x_float):
    fx = to_fixed(x_float)
    return share_secret(fx)

def shares_to_float(shares):
    return from_fixed(reconstruct(shares))

# --- simulated secure primitives (replace with ASTRA calls in real port) ---
def simulated_beaver_matmul(A_shares, B_shares):
    A = from_fixed(reconstruct(A_shares))
    B = from_fixed(reconstruct(B_shares))
    C = np.matmul(A, B)
    C_fixed = to_fixed(C)
    return share_secret(C_fixed)

def simulated_elementwise_mul_scalar(vec_shares, scalar_shares):
    v = from_fixed(reconstruct(vec_shares))
    s = from_fixed(reconstruct(scalar_shares))
    res = v * s  # supports per-row broadcasting if s shape is (N,1)
    return share_secret(to_fixed(res))

def secure_trunc(shares, k=FRAC_BITS):
    x = reconstruct(shares)
    xi_signed = x.view(np.int64)
    shifted = (xi_signed >> k).astype(np.int64)
    return share_secret(shifted.view(np.uint64))

# --- polynomial approximations used in prototype ---
def poly_sigmoid_clear(x):
    # degree-3 polynomial approx (around 0)
    return 0.5 + 0.25 * x - (x**3) / 48.0

# --- cleartext reference LEX-GNN layer (for comparison) ---
def lexgnn_layer_clear(h, W0, W1, Wself, Wdest, Psi0, Psi1, adjacency, pre_mlp):
    N = h.shape[0]
    z = h.dot(pre_mlp['W1']).dot(pre_mlp['W2'])
    p = poly_sigmoid_clear(z.flatten()).reshape(N, 1)
    base = h.dot(W0) + Psi0
    diff = h.dot(W1 - W0) + (Psi1 - Psi0)
    m = base + p * diff
    agg = np.zeros((N, m.shape[1]))
    for v in range(N):
        neighs = adjacency[v]
        if len(neighs) == 0:
            continue
        scores = np.array([max(0.0, np.dot(m[u], m[v])) for u in neighs])
        denom = scores.sum()
        if denom == 0:
            weights = np.ones_like(scores) / len(scores)
        else:
            weights = scores / denom
        agg[v] = sum(w * m[u] for w, u in zip(weights, neighs))
    base_d = agg.dot(Wdest)
    h_new = h.dot(Wself) + base_d
    return h_new, p

# --- MPC-simulated LEX-GNN layer ---
def lexgnn_layer_mpc(h_shares, W0_shares, W1_shares, Wself_shares, Wdest_shares, Psi0_shares, Psi1_shares, adjacency, pre_mlp_shares):
    N = shares_to_float(h_shares).shape[0]
    h_clear = shares_to_float(h_shares)

    # Φ_pre MLP simulated (reconstruct -> compute -> re-share)
    z = h_clear.dot(pre_mlp_shares['W1_clear']).dot(pre_mlp_shares['W2_clear'])
    p_clear = poly_sigmoid_clear(z.flatten()).reshape(N,1)
    p_shares = share_from_float(p_clear)

    # base = h @ W0 + Psi0
    base_shares = simulated_beaver_matmul(h_shares, W0_shares)
    Psi0_broadcast = np.tile(from_fixed(reconstruct(Psi0_shares)).reshape(1,-1), (N,1))
    Psi0_shares = share_from_float(Psi0_broadcast)
    base_shares = [ring_add(base_shares[i], Psi0_shares[i]) for i in range(3)]

    # diff = h @ (W1 - W0)
    Wdiff_shares = [ring_sub(W1_shares[i], W0_shares[i]) for i in range(3)]
    diff_shares = simulated_beaver_matmul(h_shares, Wdiff_shares)

    # m = base + p * diff
    p_times_diff_shares = simulated_elementwise_mul_scalar(diff_shares, p_shares)
    m_shares = [ring_add(base_shares[i], p_times_diff_shares[i]) for i in range(3)]

    # reconstruct m to compute degree-normalized attention (prototype choice)
    m_clear = shares_to_float(m_shares)
    agg = np.zeros_like(m_clear)
    for v in range(N):
        neighs = adjacency[v]
        if len(neighs) == 0:
            continue
        scores = np.array([max(0.0, np.dot(m_clear[u], m_clear[v])) for u in neighs])
        denom = scores.sum()
        if denom == 0:
            weights = np.ones_like(scores) / len(scores)
        else:
            weights = scores / denom
        agg[v] = sum(w * m_clear[u] for w, u in zip(weights, neighs))

    # project aggregated features and compute final h_new shares
    agg_shares = share_from_float(agg)
    base_d_shares = simulated_beaver_matmul(agg_shares, Wdest_shares)
    wself_h_shares = simulated_beaver_matmul(h_shares, Wself_shares)
    zero_shares = share_from_float(np.zeros_like(agg))
    h_new_shares = [ring_add(ring_add(wself_h_shares[i], base_d_shares[i]), zero_shares[i]) for i in range(3)]
    return h_new_shares, p_shares

# --- toy test driver ---
def toy_test(N=6, D=4, D_out=3, seed=42):
    np.random.seed(seed)
    h = np.random.randn(N, D).astype(np.float64)
    W0 = np.random.randn(D, D_out).astype(np.float64) * 0.5
    W1 = W0 + np.random.randn(D, D_out).astype(np.float64) * 0.1
    Wself = np.random.randn(D, D_out).astype(np.float64) * 0.2
    Wdest = np.random.randn(D_out, D_out).astype(np.float64) * 0.3
    Psi0 = np.random.randn(D_out).astype(np.float64) * 0.1
    Psi1 = Psi0 + np.random.randn(D_out).astype(np.float64) * 0.05
    pre_mlp = {'W1': np.random.randn(D, D).astype(np.float64)*0.1, 'W2': np.random.randn(D, 1).astype(np.float64)*0.1}

    adjacency = {i: [(i-1)%N, (i+1)%N] for i in range(N)}
    adjacency[0].append(2); adjacency[3].append(1)

    # cleartext
    h_new_clear, p_clear = lexgnn_layer_clear(h, W0, W1, Wself, Wdest, Psi0, Psi1, adjacency, pre_mlp)

    # share values
    h_shares = share_from_float(h)
    W0_shares = share_from_float(W0)
    W1_shares = share_from_float(W1)
    Wself_shares = share_from_float(Wself)
    Wdest_shares = share_from_float(Wdest)
    Psi0_shares = share_from_float(Psi0.reshape(1,-1))
    Psi1_shares = share_from_float(Psi1.reshape(1,-1))
    pre_mlp_shares = {'W1_clear': pre_mlp['W1'], 'W2_clear': pre_mlp['W2']}

    # simulated MPC LEX-GNN layer
    h_new_shares, p_shares = lexgnn_layer_mpc(h_shares, W0_shares, W1_shares, Wself_shares, Wdest_shares, Psi0_shares, Psi1_shares, adjacency, pre_mlp_shares)
    h_new_mpc = shares_to_float(h_new_shares)
    p_mpc = shares_to_float(p_shares).flatten()

    df_rows = []
    for i in range(N):
        # explicit scalar extraction to avoid DeprecationWarning in NumPy
        p_clear_scalar = float(p_clear[i, 0]) if p_clear.ndim > 1 else float(p_clear[i])
        df_rows.append({
            'node': i,
            'p_clear': p_clear_scalar,
            'p_mpc': float(p_mpc[i]),
            'h_new_clear_norm': float(np.linalg.norm(h_new_clear[i])),
            'h_new_mpc_norm': float(np.linalg.norm(h_new_mpc[i]))
        })
    df = pd.DataFrame(df_rows)
    print(df)
    return {'df': df, 'h_new_clear': h_new_clear, 'h_new_mpc': h_new_mpc, 'p_clear': p_clear.flatten(), 'p_mpc': p_mpc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated ASTRA-style MPC LEX-GNN toy test")
    parser.add_argument("--N", type=int, default=6, help="number of nodes")
    parser.add_argument("--D", type=int, default=4, help="feature dim")
    parser.add_argument("--Dout", type=int, default=3, help="output dim")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    toy_test(N=args.N, D=args.D, D_out=args.Dout, seed=args.seed)
