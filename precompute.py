# precompute.py
# Offline phase: Generates all required Beaver triples and saves them to files.
import numpy as np
import pickle

# Default fallback
N=6; D=4; Dout=3
# MLP Hidden Dimension
H_dim = 8 

rand_bound = 3
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

def share_uint64(x_uint):
    x = np.asarray(x_uint, dtype=np.uint64)
    rand_share_bound = 1 << 20
    s0 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s1 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s2_list = []
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        xi = int(x[idx]); v0 = int(s0[idx]); v1 = int(s1[idx])
        s2_val = (xi - v0 - v1) & ((1 << 64) - 1)
        s2_list.append(s2_val)
    s2 = np.array(s2_list, dtype=np.uint64).reshape(x.shape)
    return [s0, s1, s2]

def gen_triples(shape, op_type='matmul'):
    """Generates and returns (a_sh, b_sh, c_sh) for the given operation."""
    if op_type == 'matmul':
        M, K, N_shape = shape
        a = np.random.randint(-rand_bound, rand_bound, size=(M,K)).astype(np.int64)
        b = np.random.randint(-rand_bound, rand_bound, size=(K,N_shape)).astype(np.int64)
        a_f = (a * SCALE).astype(np.int64)
        b_f = (b * SCALE).astype(np.int64)
        prod_ab = a_f.dot(b_f).astype(np.int64)
        c_f = np.right_shift(prod_ab, FRAC_BITS).astype(np.int64)
    else: # hadamard
        M, N_shape = shape
        a = np.random.randint(-rand_bound, rand_bound, size=(M,N_shape)).astype(np.int64)
        b = np.random.randint(-rand_bound, rand_bound, size=(M,N_shape)).astype(np.int64)
        a_f = (a * SCALE).astype(np.int64)
        b_f = (b * SCALE).astype(np.int64)
        prod_ab = (a_f * b_f).astype(np.int64)
        c_f = np.right_shift(prod_ab, FRAC_BITS).astype(np.int64)

    a_u, b_u, c_u = a_f.view(np.uint64), b_f.view(np.uint64), c_f.view(np.uint64)
    return share_uint64(a_u), share_uint64(b_u), share_uint64(c_u)

if __name__ == "__main__":
    print("--- Starting Offline Phase: Triple Generation ---")
    np.random.seed(42) 
    
    try:
        print("Loading dimensions from mini_fraud_data.pkl...")
        with open('mini_fraud_data.pkl', 'rb') as f:
            real_data = pickle.load(f)
        N = real_data['N']
        D = real_data['D']
        Dout = real_data['Dout']
        print(f"Updated Dimensions: N={N}, D={D}, Dout={Dout}")
    except FileNotFoundError:
        print(" Warning: mini_fraud_data.pkl not found. Using defaults (N=6).")

    node_data = [{}, {}, {}] 
    
    # List of all triples to generate
    triples_to_gen = [
        # --- Existing GNN triples ---
        ('base', (N,D,Dout), 'matmul'),
        ('diff', (N,D,Dout), 'matmul'),
        ('pdiff', (N,Dout), 'hadamard'),
        ('based', (N,Dout,Dout), 'matmul'),
        ('wselfh', (N,D,Dout), 'matmul'),
        ('dot', (1,Dout,1), 'matmul'),
        ('score_sq', (1,1), 'hadamard'),
        ('psi_p', (N, D), 'hadamard'),
        
        # --- MLP (Eq 1) triples ---
        ('mlp1', (N, D, H_dim), 'matmul'),
        ('mlp_act', (N, H_dim), 'hadamard'),
        ('mlp2', (N, H_dim, 1), 'matmul'),
        
        # --- Loss (Eq 2) triples ---
        ('loss_term', (N, 1), 'hadamard'),

        # --- Destination Update (Eq 7) ---
        ('agg_Wd0', (N, Dout, Dout), 'matmul'),
        ('agg_Wdiff', (N, Dout, Dout), 'matmul'),
        ('dest_term2', (N, Dout), 'hadamard'),

        # --- NEW: Eq 8 (Classifier MLP & Loss) ---
        # Layer 1: h_new(N, Dout) @ W_cls1(Dout, H)
        ('cls1', (N, Dout, H_dim), 'matmul'),
        # Activation: (N, H)
        ('cls_act', (N, H_dim), 'hadamard'),
        # Layer 2: hidden(N, H) @ W_cls2(H, 1)
        ('cls2', (N, H_dim, 1), 'matmul'),
        # Loss terms: (N, 1)
        ('cls_loss', (N, 1), 'hadamard')
    ]
    
    for prefix, shape, op_type in triples_to_gen:
        print(f"Generating triples for [ {prefix} ] shape {shape}...")
        a_sh, b_sh, c_sh = gen_triples(shape, op_type)
        
        for nid in range(3):
            node_data[nid][f'a_{prefix}'] = a_sh[nid]
            node_data[nid][f'b_{prefix}'] = b_sh[nid]
            node_data[nid][f'c_{prefix}'] = c_sh[nid]

    # Save the data to files
    for nid in range(3):
        filename = f'node{nid}_offline.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(node_data[nid], f)
        print(f" Saved offline data for node {nid} to {filename}")
        
    print("--- Offline Phase Complete ---")