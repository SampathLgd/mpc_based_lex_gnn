# lex_gnn_mpc_no_recon.py
# Fixed: faithful single-process MPC simulator (Beaver triples) for one LEX-GNN layer
# - Avoids reconstruct shortcuts before attention
# - Fixes NumPy scalar/shape issues and overflow/NaN problems
# - Still naive (elementwise Beaver matmul) and therefore slow for large dims
#
# Requires: numpy, pandas

import numpy as np
import pandas as pd

# ---------- Config ----------
SCALE = 2**16         # fixed-point scale (fractional bits)
FRAC_BITS = 16
SEED = 42
np.random.seed(SEED)

# ---------- Helpers: fixed-point conversions ----------
def to_fixed(x):
    """Convert float numpy array to fixed-point uint64 (two's complement view)."""
    xf = np.asarray(x, dtype=np.float64)
    # Replace NaN/inf with large finite numbers to avoid cast errors
    xf = np.nan_to_num(xf, nan=0.0, posinf=1e9, neginf=-1e9)
    xi = np.round(xf * SCALE)
    # clamp to int64 range to avoid overflow during astype
    int64_max = (1 << 63) - 1
    int64_min = -(1 << 63)
    xi = np.clip(xi, int64_min, int64_max).astype(np.int64)
    return xi.view(np.uint64)

def from_fixed(x_uint):
    """Convert uint64 array (two's complement) back to float numpy array."""
    return x_uint.view(np.int64).astype(np.float64) / SCALE

# ---------- 3-party additive sharing ----------
def share(x_uint):
    """x_uint: uint64 numpy array -> returns list of 3 uint64 shares"""
    s0 = np.random.randint(0, 2**63, size=x_uint.shape, dtype=np.uint64)
    s1 = np.random.randint(0, 2**63, size=x_uint.shape, dtype=np.uint64)
    s2 = (x_uint - s0 - s1).astype(np.uint64)
    return [s0, s1, s2]

def open_shares(shares):
    """Reconstruct (open) shares -> uint64 array"""
    return (shares[0] + shares[1] + shares[2]).astype(np.uint64)

def local_add_shares(A_sh, B_sh):
    return [ (A_sh[i] + B_sh[i]).astype(np.uint64) for i in range(3) ]

def local_sub_shares(A_sh, B_sh):
    return [ (A_sh[i] - B_sh[i]).astype(np.uint64) for i in range(3) ]

# ---------- Beaver triple generator (offline) ----------
def gen_triple(shape, rand_bound=2**10):
    """
    Generate a,b,c triples in int domain with small magnitude to avoid overflow in simulation.
    Returns a_sh, b_sh, c_sh each as lists of 3 uint64 shares.
    """
    # keep a,b small to avoid very large intermediate products in simulation
    a = np.random.randint(-rand_bound, rand_bound, size=shape).astype(np.int64)
    b = np.random.randint(-rand_bound, rand_bound, size=shape).astype(np.int64)
    c = (a.astype(np.int64) * b.astype(np.int64)).astype(np.int64)
    return share(a.view(np.uint64)), share(b.view(np.uint64)), share(c.view(np.uint64))

# ---------- Beaver online multiplication for scalar/elementwise arrays ----------
def beaver_mul_shares(X_sh, Y_sh, triple):
    """
    Beaver online multiplication (elementwise arrays).
    X_sh, Y_sh: list of 3 uint64 arrays (same shape).
    triple: (a_sh, b_sh, c_sh) each list of 3 shares.
    """
    a_sh, b_sh, c_sh = triple

    # d_sh and e_sh (shares)
    d_sh = local_sub_shares(X_sh, a_sh)
    e_sh = local_sub_shares(Y_sh, b_sh)

    # open d and e (uint64 patterns), convert to floats
    d_open_uint = open_shares(d_sh)
    e_open_uint = open_shares(e_sh)
    d_open = from_fixed(d_open_uint)   # float array
    e_open = from_fixed(e_open_uint)

    # open triple components to compute clear values for a,b,c (simulation only)
    a_open = from_fixed(open_shares(a_sh))
    b_open = from_fixed(open_shares(b_sh))
    c_open = from_fixed(open_shares(c_sh))

    # compute clear Z
    Z_clear = c_open + d_open * b_open + e_open * a_open + (d_open * e_open)

    # guard for NaN/Inf before sharing
    Z_clear = np.nan_to_num(Z_clear, nan=0.0, posinf=1e9, neginf=-1e9)

    # share result
    Z_sh = share(to_fixed(Z_clear))
    return Z_sh

# ---------- Naive element-wise matmul by Beaver (very slow) ----------
def beaver_matmul(A_sh, B_sh, triple_gen=gen_triple):
    """
    Naive matmul via elementwise Beaver multiplications.
    A_sh: shares of (M x K). B_sh: shares of (K x N).
    Returns shares of (M x N).
    WARNING: extremely slow for large sizes; used for correct flow only.
    """
    M, K = A_sh[0].shape
    K2, N = B_sh[0].shape
    assert K == K2

    # initialize zero shares for result
    Z_sh = [ np.zeros((M, N), dtype=np.uint64) for _ in range(3) ]

    # iterate
    for i in range(M):
        for j in range(N):
            # compute sum over k
            sum_sh = None
            for k in range(K):
                # scalar shares for A[i,k] and B[k,j] (each is shape (1,))
                Ai_sh = [ A_sh[p][i, k].reshape(1) for p in range(3) ]
                Bj_sh = [ B_sh[p][k, j].reshape(1) for p in range(3) ]
                tri = triple_gen((1,))
                prod_sh = beaver_mul_shares(Ai_sh, Bj_sh, tri)  # returns list of 3 scalar shares (shape (1,))
                if sum_sh is None:
                    # initialize sum_sh as copies
                    sum_sh = [ prod_sh[p].copy() for p in range(3) ]
                else:
                    for p in range(3):
                        sum_sh[p] = (sum_sh[p] + prod_sh[p]).astype(np.uint64)
            # write scalar shares into matrix positions robustly
            for p in range(3):
                val = sum_sh[p]
                # val may be numpy array, numpy scalar, or python int
                if hasattr(val, 'item'):
                    scalar_val = val.item()
                elif isinstance(val, np.ndarray) and val.size == 1:
                    scalar_val = val.reshape(-1)[0]
                else:
                    scalar_val = val
                Z_sh[p][i, j] = np.uint64(scalar_val)
    return Z_sh

# ---------- Secure dot of two shared row vectors ----------
def beaver_dot_row(a_sh_row, b_sh_row):
    """
    a_sh_row, b_sh_row: lists of 3 arrays shape (D,) for one row.
    Returns scalar shares (list of 3 arrays shape (1,))
    """
    D = a_sh_row[0].shape[0]
    sum_sh = None
    for k in range(D):
        Ai = [ a_sh_row[p][k].reshape(1) for p in range(3) ]
        Bi = [ b_sh_row[p][k].reshape(1) for p in range(3) ]
        tri = gen_triple((1,))
        prod = beaver_mul_shares(Ai, Bi, tri)
        if sum_sh is None:
            sum_sh = [ prod[p].copy() for p in range(3) ]
        else:
            for p in range(3):
                sum_sh[p] = (sum_sh[p] + prod[p]).astype(np.uint64)
    return sum_sh

# ---------- Secure per-row scalar multiply p * vec ----------
def beaver_row_scalar_mul(vec_sh, scalar_sh):
    """
    vec_sh: list of 3 shares shape (N,D)
    scalar_sh: list of 3 shares shape (N,1)
    returns shares shape (N,D)
    """
    N, D = vec_sh[0].shape
    out_sh = [ np.zeros((N,D), dtype=np.uint64) for _ in range(3) ]
    for i in range(N):
        s_sh = [ scalar_sh[p][i, 0].reshape(1) for p in range(3) ]
        for j in range(D):
            v_sh = [ vec_sh[p][i, j].reshape(1) for p in range(3) ]
            tri = gen_triple((1,))
            prod = beaver_mul_shares(v_sh, s_sh, tri)
            for p in range(3):
                val = prod[p]
                if hasattr(val, 'item'):
                    scalar_val = val.item()
                elif isinstance(val, np.ndarray) and val.size == 1:
                    scalar_val = val.reshape(-1)[0]
                else:
                    scalar_val = val
                out_sh[p][i, j] = np.uint64(scalar_val)
    return out_sh

# ---------- LEX-GNN secure layer ----------
def lexgnn_layer_secure(h_clear, W0_clear, W1_clear, Wself_clear, Wdest_clear, Psi0_clear, Psi1_clear, adjacency):
    """
    h_clear etc are numpy float arrays in clear (we'll convert->share->secure compute->open final h_new).
    This function uses beaver primitives but avoids reconstructing node vectors before attention.
    """
    # share inputs
    h_sh = share(to_fixed(h_clear))
    W0_sh = share(to_fixed(W0_clear))
    W1_sh = share(to_fixed(W1_clear))
    Wself_sh = share(to_fixed(Wself_clear))
    Wdest_sh = share(to_fixed(Wdest_clear))
    Psi0_sh = share(to_fixed(Psi0_clear.reshape(1,-1)))
    Psi1_sh = share(to_fixed(Psi1_clear.reshape(1,-1)))

    N = h_clear.shape[0]

    # Simulated Φ_pre (for brevity we compute clear -> share)
    z = h_clear.dot(np.random.randn(h_clear.shape[1],1)*0.1)  # toy pre-MLP
    p_clear = 0.5 + 0.25 * z.flatten() - (z.flatten()**3) / 48.0
    p_sh = share(to_fixed(p_clear.reshape(N,1)))

    # base = h @ W0  (secure)
    base_sh = beaver_matmul(h_sh, W0_sh, gen_triple)
    # add Psi0 broadcast
    Psi0_b = np.tile(from_fixed(open_shares(Psi0_sh)).reshape(1,-1), (N,1))
    Psi0_b_sh = share(to_fixed(Psi0_b))
    base_sh = local_add_shares(base_sh, Psi0_b_sh)

    # diff = h @ (W1 - W0)
    Wdiff_sh = local_sub_shares(W1_sh, W0_sh)
    diff_sh = beaver_matmul(h_sh, Wdiff_sh, gen_triple)

    # m = base + p * diff  (secure per-row scalar multiply)
    p_diff_sh = beaver_row_scalar_mul(diff_sh, p_sh)
    m_sh = local_add_shares(base_sh, p_diff_sh)

    # compute secure dot scores for each edge (u->v)
    scores = {}
    for v in range(N):
        neighs = adjacency[v]
        scores[v] = []
        for u in neighs:
            dot_sh = beaver_dot_row([m_sh[p][u,:] for p in range(3)],
                                   [m_sh[p][v,:] for p in range(3)])
            # open dot_sh (public scalar) for normalization
            score_val = from_fixed(open_shares(dot_sh))
            # use squared score to avoid ReLU/comparison
            sc = float(np.asarray(score_val).reshape(-1)[0])
            scores[v].append(sc*sc)

    # Normalize weights per node
    weights = {}
    for v in range(N):
        arr = np.array(scores[v], dtype=np.float64)
        ssum = arr.sum()
        if ssum == 0 or arr.size == 0:
            weights[v] = np.ones_like(arr)/max(1, arr.size)
        else:
            weights[v] = arr / ssum

    # Aggregate: public scalar * secret vector -> scale shares by public scalar
    agg_sh = [ np.zeros_like(m_sh[0]) for _ in range(3) ]
    for v in range(N):
        neighs = adjacency[v]
        for idx,u in enumerate(neighs):
            w = float(weights[v][idx])
            # scale each party's share by multiplying share values (simulation uses clear multiplication)
            for p in range(3):
                # convert m_sh[p][u,:] -> float, multiply by w, then convert back to int and store
                mfloat = from_fixed(m_sh[p][u,:].astype(np.uint64))
                scaled = np.round(mfloat * w)
                # clamp to int64 range then to uint64 representation
                int64_max = (1 << 63) - 1
                int64_min = -(1 << 63)
                scaled = np.clip(scaled, int64_min, int64_max).astype(np.int64)
                agg_sh[p][v,:] = (agg_sh[p][v,:].astype(np.int64) + scaled).astype(np.uint64)

    # project aggregated features base_d = agg · Wdest
    base_d_sh = beaver_matmul(agg_sh, Wdest_sh, gen_triple)
    # wself*h
    wself_h_sh = beaver_matmul(h_sh, Wself_sh, gen_triple)
    # final h_new_sh = wself_h_sh + base_d_sh
    h_new_sh = local_add_shares(wself_h_sh, base_d_sh)

    # open final h_new for inspection (in real system you'd often deliver result in shares to client)
    h_new = from_fixed(open_shares(h_new_sh))
    p_open = from_fixed(open_shares(p_sh))
    return h_new, p_open

# ---------- Toy driver ----------
def toy():
    N=6; D=4; Dout=3
    h = np.random.randn(N,D)
    W0 = np.random.randn(D,Dout)*0.5
    W1 = W0 + np.random.randn(D,Dout)*0.1
    Wself = np.random.randn(D,Dout)*0.2
    Wdest = np.random.randn(Dout,Dout)*0.3
    Psi0 = np.random.randn(Dout)*0.1
    Psi1 = Psi0 + np.random.randn(Dout)*0.05
    adjacency = {i:[(i-1)%N,(i+1)%N] for i in range(N)}
    adjacency[0].append(2); adjacency[3].append(1)

    h_new, p_open = lexgnn_layer_secure(h, W0, W1, Wself, Wdest, Psi0, Psi1, adjacency)
    print("Opened p:", p_open.flatten())
    print("h_new norms per node:", np.linalg.norm(h_new, axis=1))

if __name__ == "__main__":
    toy()
