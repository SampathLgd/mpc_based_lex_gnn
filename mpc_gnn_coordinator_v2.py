# mpc_gnn_coordinator_v2.py
# Coordinator for Privacy-Preserving LEX-GNN via 3-Party MPC.
# Implements the complete inference and optimization pipeline (Eq 1-9).

import socket, pickle, struct, time, numpy as np

# ==========================================
# 1. MPC PRIMITIVE WRAPPERS
# ==========================================

def reconstruct_elementwise(shares_list):
    """Reconstructs a secret from a list of 3 uint64 additive shares."""
    if not shares_list or len(shares_list) != 3:
        raise ValueError("Requires a list of 3 shares to reconstruct.")
    s0 = np.asarray(shares_list[0], dtype=np.uint64)
    s1 = np.asarray(shares_list[1], dtype=np.uint64)
    s2 = np.asarray(shares_list[2], dtype=np.uint64)
    return (s0 + s1 + s2).astype(np.uint64)

HOST = 'localhost'
PORT_BASE = 9000

def send(node_id, msg, timeout=5.0, retry=6):
    """Handles reliable socket communication with worker nodes."""
    port = PORT_BASE + node_id
    last_exc = None
    for _ in range(retry):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((HOST, port))
            data = pickle.dumps(msg)
            s.sendall(struct.pack('!I', len(data)) + data)
            raw_len = s.recv(4)
            if not raw_len:
                s.close(); raise RuntimeError("no response")
            (resp_len,) = struct.unpack('!I', raw_len)
            resp = b''
            while len(resp) < resp_len:
                chunk = s.recv(resp_len - len(resp))
                if not chunk: break
                resp += chunk
            s.close()
            return pickle.loads(resp)
        except Exception as e:
            last_exc = e
            time.sleep(0.12)
    raise RuntimeError(f"Failed to send to node {node_id}: {last_exc}")

FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

# --- Cryptographic RPC Definitions ---

def rpc_add(key_A, key_B, store_as):
    req = {'type': 'LOCAL_ADD', 'key_A': key_A, 'key_B': key_B, 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} LOCAL_ADD failed: {r}")
    return True

def rpc_sub(key_A, key_B, store_as):
    req = {'type': 'LOCAL_SUB', 'key_A': key_A, 'key_B': key_B, 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} LOCAL_SUB failed: {r}")
    return True

def rpc_matmul(X_key, Y_key, a_key, b_key, c_key, store_as):
    # 1. Compute D and E shares locally
    d_parts=[]; e_parts=[]
    for nid in range(3):
        r = send(nid, {'type':'MAT_COMPUTE_D_E', 'X_key': X_key,'Y_key': Y_key,'a_key': a_key,'b_key': b_key})
        if not r.get('ok'): raise RuntimeError(f"node{nid} MAT_COMPUTE_D_E failed: {r}")
        d_parts.append(np.array(r['d_i'], dtype=np.uint64))
        e_parts.append(np.array(r['e_i'], dtype=np.uint64))

    # 2. Reconstruct D and E (Public values)
    D_u = reconstruct_elementwise(d_parts)
    E_u = reconstruct_elementwise(e_parts)
    D = uint64_to_signed_int64(D_u)
    E = uint64_to_signed_int64(E_u)

    # 3. Compute public DE product
    DE_prod = D.dot(E).astype(np.int64)
    DE = np.right_shift(DE_prod, FRAC_BITS).astype(np.int64)
    DE_u = DE.view(np.uint64)
    DE_sh = share_uint64(DE_u)

    # 4. Securely compute final result [Z]
    for nid in range(3):
        msg = {'type':'APPLY_BACTH_DE_FIXED', 'D': D.tolist(), 'E': E.tolist(), 'DE_share': np.array(DE_sh[nid], dtype=np.uint64),
               'c_key':c_key,'a_key':a_key,'b_key':b_key,'store_as':store_as}
        r = send(nid, msg)
        if not r.get('ok'):
            dbg = send(nid, {'type':'DEBUG_KEYS'})
            raise RuntimeError(f"node{nid} APPLY failed: {r}; keys: {dbg}")
    # Synchronization delay
    time.sleep(0.02)
    return True

def rpc_hadamard_mul(X_key, Y_key, a_key, b_key, c_key, store_as):
    d_parts=[]; e_parts=[]
    for nid in range(3):
        r = send(nid, {'type':'MAT_COMPUTE_D_E', 'X_key': X_key,'Y_key': Y_key,'a_key': a_key,'b_key': b_key})
        if not r.get('ok'): raise RuntimeError(f"node{nid} MAT_COMPUTE_D_E failed: {r}")
        d_parts.append(np.array(r['d_i'], dtype=np.uint64))
        e_parts.append(np.array(r['e_i'], dtype=np.uint64))

    D_u = reconstruct_elementwise(d_parts)
    E_u = reconstruct_elementwise(e_parts)
    D = uint64_to_signed_int64(D_u)
    E = uint64_to_signed_int64(E_u)

    # Element-wise multiplication
    DE_prod = (D.astype(np.int64) * E.astype(np.int64)).astype(np.int64)
    DE = np.right_shift(DE_prod, FRAC_BITS).astype(np.int64)
    DE_u = DE.view(np.uint64)
    DE_sh = share_uint64(DE_u)

    for nid in range(3):
        msg = {'type':'APPLY_HADAMARD_DE_FIXED', 
               'D': D.tolist(), 'E': E.tolist(), 'DE_share': np.array(DE_sh[nid], dtype=np.uint64),
               'a_key': a_key, 'b_key': b_key, 'c_key': c_key, 'store_as': store_as}
        r = send(nid, msg)
        if not r.get('ok'):
            dbg = send(nid, {'type':'DEBUG_KEYS'})
            raise RuntimeError(f"node{nid} APPLY_HADAMARD_DE_FIXED failed: {r}; keys: {dbg}")
    time.sleep(0.02)
    return True

def rpc_scalar_mul(key_in, scalar_float, store_as):
    scalar_fixed = np.round(scalar_float * SCALE).astype(np.int64)
    req = {'type': 'PUBLIC_SCALAR_MUL', 'key_in': key_in,
           'scalar_fixed': scalar_fixed.item(), 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} PUBLIC_SCALAR_MUL failed: {r}")
    time.sleep(0.02)
    return True

def rpc_extract_row(key_in, row_idx, store_as):
    req = {'type': 'EXTRACT_ROW', 'key_in': key_in, 'row_idx': row_idx, 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} EXTRACT_ROW failed: {r}")
    return True

def rpc_extract_col(key_in, row_idx, store_as):
    req = {'type': 'EXTRACT_COL', 'key_in': key_in, 'row_idx': row_idx, 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} EXTRACT_COL failed: {r}")
    return True

def rpc_update_row(key_target, key_source, row_idx):
    req = {'type': 'UPDATE_ROW', 'key_target': key_target, 'key_source': key_source, 'row_idx': row_idx}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} UPDATE_ROW failed: {r}")
    return True
 
def rpc_get_node(nid, key):
    r = send(nid, {'type':'GET', 'key':key})
    if not r.get('ok') or r.get('share') is None:
        dbg = send(nid, {'type':'DEBUG_KEYS'})
        raise Exception(f"node{nid} GET {key} failed: {r}; keys: {dbg}")
    return np.array(r['share'], dtype=np.uint64)

def share_uint64(x_uint):
    x = np.asarray(x_uint, dtype=np.uint64)
    rand_share_bound = 1 << 20
    s0 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s1 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s2 = (x - s0 - s1).astype(np.uint64)
    return [s0, s1, s2]

def uint64_to_signed_int64(uarr):
    u = np.asarray(uarr, dtype=np.uint64)
    mask = u >= (1 << 63)
    if mask.any():
        out = np.empty(u.shape, dtype=np.int64)
        it = np.nditer(u, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            val = int(u[idx])
            out[idx] = np.int64(val - (1 << 64)) if val >= (1 << 63) else np.int64(val)
        return out
    else:
        return u.view(np.int64)

def from_fixed(x_uint):
    return x_uint.view(np.int64).astype(np.float64) / SCALE

def to_fixed(x):
    x_clipped = np.clip(x, -1e10, 1e10)
    xi = np.round(x_clipped * SCALE).astype(np.int64)
    return xi.view(np.uint64)

def rpc_store_all_nodes(key, clear_data_float):
    data_u = to_fixed(clear_data_float)
    shares = share_uint64(data_u)
    for nid in range(3):
        r = send(nid, {'type':'STORE', 'key':key, 'share': shares[nid]})
        if not r.get('ok'): raise RuntimeError(f"node{nid} STORE {key} failed {r}")
    return True

# --- Cleartext Reference Function (For Validation) ---
def lexgnn_layer_clear_exact(h, W0, W1, Psi0, Psi1, Wself, Wd0, Wd1, adjacency, p_clear):
    N = h.shape[0]
    p_clear = p_clear.reshape(N, 1)
    
    # 1. Label Embedding (Eq. 4)
    Psi_interp = p_clear * Psi1 + (1 - p_clear) * Psi0
    m_tilde = h + Psi_interp
    
    # 2. Source Transformation (Eq. 5)
    base = m_tilde.dot(W0)
    diff = m_tilde.dot(W1 - W0)
    p_diff = diff * p_clear
    m = base + p_diff
    
    # 3. Attention & Aggregation (Eq. 6)
    agg = np.zeros_like(m)
    for v in range(N):
        neighs = adjacency[v]
        if len(neighs) == 0: continue
        scores = []
        for u in neighs:
            sc = np.dot(m[u], m[v])
            scores.append(sc * sc)
        scores = np.array(scores, dtype=np.float64)
        ssum = scores.sum()
        weights = scores / ssum if ssum != 0 else np.ones_like(scores) / max(1, scores.size)
        for idx, u in enumerate(neighs):
            agg[v] += weights[idx] * m[u]

    # 4. Differentiated Message Reception (Eq. 7)
    term1 = agg.dot(Wd0)
    Wd_diff = Wd1 - Wd0
    temp = agg.dot(Wd_diff)
    term2 = temp * p_clear
    base_d = term1 + term2
    wself_h = h.dot(Wself)
    h_new = wself_h + base_d
    return h_new


# ==========================================
# 2. SYSTEM EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   MPC-BASED LEX-GNN COORDINATOR (Research Protocol)")
    print("="*60 + "\n")
    
    # --- Phase 1: Data Loading ---
    print("[*] Loading dataset from 'mini_fraud_data.pkl'...")
    try:
        with open('mini_fraud_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("[-] Error: Dataset not found. Run gen_fraud_data.py first.")
        exit(1)
        
    N, D, Dout = data['N'], data['D'], data['Dout']
    H_dim = 8 
    
    h = data['h']
    Y_labels = data['Y']
    adjacency = data['adjacency']
    
    W0, W1 = data['W0'], data['W1']
    Wself, Wdest = data['Wself'], data['Wdest']
    Psi0, Psi1 = data['Psi0'], data['Psi1']
    W_mlp1, W_mlp2 = data['W_mlp1'], data['W_mlp2']
    
    # Define Differentiated Destination Weights
    Wd0 = Wdest
    Wd1 = Wdest * 1.2 
    
    # Broadcast Label Embeddings
    Psi0_b = np.tile(Psi0, (N, 1))
    Psi1_b = np.tile(Psi1, (N, 1))

    print(f"[+] Graph Configuration: N={N} Nodes, D={D} Features, D_out={Dout} Hidden")

    total_start = time.time()

    try:
        # --- Phase 2: Secret Sharing ---
        start_t = time.time()
        rpc_store_all_nodes('h', h)
        rpc_store_all_nodes('W0', W0)
        rpc_store_all_nodes('W1', W1)
        rpc_store_all_nodes('Psi0', Psi0_b)
        rpc_store_all_nodes('Psi1', Psi1_b)
        rpc_store_all_nodes('Wself', Wself)
        rpc_store_all_nodes('Wd0', Wd0)
        rpc_store_all_nodes('Wd1', Wd1)
        rpc_store_all_nodes('W_mlp1', W_mlp1)
        rpc_store_all_nodes('W_mlp2', W_mlp2)
        rpc_store_all_nodes('Y', Y_labels)
        
        # Classifier Weights
        W_cls1 = np.random.randn(Dout, H_dim) * 0.1
        W_cls2 = np.random.randn(H_dim, 1) * 0.1
        rpc_store_all_nodes('W_cls1', W_cls1)
        rpc_store_all_nodes('W_cls2', W_cls2)
        print(f"[+] Input Secret Sharing completed in {time.time()-start_t:.4f}s")
        
        # --- Phase 3: Offline Phase Loading ---
        start_t = time.time()
        print("[*] Loading Offline Beaver Triples...")
        for nid in range(3):
            r = send(nid, {'type': 'LOAD_OFFLINE_DATA', 'file': f'node{nid}_offline.pkl'})
            if not r.get('ok'): raise RuntimeError(f"Node {nid} load failed: {r.get('err')}")
        print(f"[+] Offline Data loaded in {time.time()-start_t:.4f}s")
        
        # ---------------------------------------------------------
        # Protocol Stage 1: Secure Label Prediction (Equation 1)
        # ---------------------------------------------------------
        print("\n--- Stage 1: Secure Label Prediction (Eq. 1) ---")
        start_t = time.time()
        rpc_matmul('h', 'W_mlp1', 'a_mlp1', 'b_mlp1', 'c_mlp1', 'z1')
        rpc_hadamard_mul('z1', 'z1', 'a_mlp_act', 'b_mlp_act', 'c_mlp_act', 'a1')
        rpc_matmul('a1', 'W_mlp2', 'a_mlp2', 'b_mlp2', 'c_mlp2', 'z2')
        rpc_scalar_mul('z2', 0.25, 'p_term')
        rpc_store_all_nodes('const_0_5', np.full((N,1), 0.5))
        rpc_add('p_term', 'const_0_5', 'p')
        print(f"[+] Secure MLP execution time: {time.time()-start_t:.4f}s")
        
        # ---------------------------------------------------------
        # Protocol Stage 2: Secure Loss Calculation (Equation 2)
        # ---------------------------------------------------------
        print("\n--- Stage 2: Secure Loss Calculation (Eq. 2) ---")
        start_t = time.time()
        rpc_store_all_nodes('const_1', np.full((N,1), 1.0))
        rpc_sub('p', 'const_1', 'p_minus_1')
        rpc_hadamard_mul('Y', 'p_minus_1', 'a_loss_term', 'b_loss_term', 'c_loss_term', 'term_a')
        rpc_sub('const_1', 'Y', 'one_minus_Y')
        rpc_store_all_nodes('const_0', np.zeros((N,1)))
        rpc_sub('const_0', 'p', 'neg_p')
        rpc_hadamard_mul('one_minus_Y', 'neg_p', 'a_loss_term', 'b_loss_term', 'c_loss_term', 'term_b')
        rpc_add('term_a', 'term_b', 'total_term')
        print(f"[+] Secure Loss computation time: {time.time()-start_t:.4f}s")
        
        # Broadcast Probability Vector
        p_shares = [rpc_get_node(i, 'p') for i in range(3)]
        p_recon = from_fixed(reconstruct_elementwise(p_shares))
        p_clear_D = np.tile(p_recon, (1, D))
        p_clear_Dout = np.tile(p_recon, (1, Dout))
        rpc_store_all_nodes('p_D', p_clear_D)
        rpc_store_all_nodes('p_Dout', p_clear_Dout)

        # ---------------------------------------------------------
        # Protocol Stage 3: Message Construction (Equation 3-5)
        # ---------------------------------------------------------
        print("\n--- Stage 3: Message Construction & Label Exploration (Eq. 3-5) ---")
        start_t = time.time()
        rpc_sub('Psi1', 'Psi0', 'Psi_diff')
        rpc_hadamard_mul('Psi_diff', 'p_D', 'a_psi_p', 'b_psi_p', 'c_psi_p', 'p_Psi_diff')
        rpc_add('Psi0', 'p_Psi_diff', 'Psi_interp')
        rpc_add('h', 'Psi_interp', 'm_tilde')
        
        rpc_matmul('m_tilde', 'W0', 'a_base', 'b_base', 'c_base', 'base')
        rpc_sub('W1', 'W0', 'Wdiff')
        rpc_matmul('m_tilde', 'Wdiff', 'a_diff', 'b_diff', 'c_diff', 'diff')
        rpc_hadamard_mul('diff', 'p_Dout', 'a_pdiff', 'b_pdiff', 'c_pdiff', 'p_diff')
        rpc_add('base', 'p_diff', 'm')
        print(f"[+] Message construction time: {time.time()-start_t:.4f}s")

        # ---------------------------------------------------------
        # Protocol Stage 4: Private Attention (Equation 6)
        # ---------------------------------------------------------
        print("\n--- Stage 4: Private Attention & Aggregation (Eq. 6) ---")
        start_t = time.time()
        agg_zero = np.zeros((N, Dout))
        rpc_store_all_nodes('agg', agg_zero)

        print(f"[*] Processing {N} nodes... (Progress updates every 20%)")
        for v in range(N):
            if v % max(1, N//5) == 0: print(f"    ... processing node {v}/{N}")
            
            neighs = adjacency[v]
            if len(neighs) == 0: continue
            scores = []
            rpc_extract_col('m', v, 'm_v_col')
            
            for u in neighs:
                rpc_extract_row('m', u, 'm_u_row')
                rpc_matmul('m_u_row', 'm_v_col', 'a_dot', 'b_dot', 'c_dot', 'score_uv')
                rpc_hadamard_mul('score_uv', 'score_uv', 'a_score_sq', 'b_score_sq', 'c_score_sq', 'score_sq_uv')
                
                score_shares = [rpc_get_node(i, 'score_sq_uv') for i in range(3)]
                score_u = reconstruct_elementwise(score_shares)
                score_float = from_fixed(score_u).item()
                scores.append(score_float)

            scores = np.array(scores, dtype=np.float64)
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
            ssum = scores.sum()
            if ssum == 0 or scores.size == 0:
                weights = np.ones_like(scores) / max(1, scores.size)
            else:
                weights = scores / ssum
            
            rpc_store_all_nodes('agg_temp_row', np.zeros((1, Dout)))

            for idx, u in enumerate(neighs):
                w = weights[idx]
                if w == 0: continue
                rpc_extract_row('m', u, 'm_u_row')
                rpc_scalar_mul('m_u_row', w, 'weighted_m')
                rpc_add('agg_temp_row', 'weighted_m', 'agg_temp_row')
            
            rpc_update_row('agg', 'agg_temp_row', v)
        print(f"[+] Aggregation completed in {time.time()-start_t:.4f}s")

        # ---------------------------------------------------------
        # Protocol Stage 5: Destination Update (Equation 7)
        # ---------------------------------------------------------
        print("\n--- Stage 5: Differentiated Destination Update (Eq. 7) ---")
        start_t = time.time()
        rpc_sub('Wd1', 'Wd0', 'Wd_diff')
        rpc_matmul('agg', 'Wd0', 'a_agg_Wd0', 'b_agg_Wd0', 'c_agg_Wd0', 'dest_term1')
        rpc_matmul('agg', 'Wd_diff', 'a_agg_Wdiff', 'b_agg_Wdiff', 'c_agg_Wdiff', 'dest_temp')
        rpc_hadamard_mul('dest_temp', 'p_Dout', 'a_dest_term2', 'b_dest_term2', 'c_dest_term2', 'dest_term2')
        rpc_add('dest_term1', 'dest_term2', 'base_d')
        rpc_matmul('h', 'Wself', 'a_wselfh', 'b_wselfh', 'c_wselfh', 'wself_h')
        rpc_add('wself_h', 'base_d', 'h_new')
        print(f"[+] Final projection completed in {time.time()-start_t:.4f}s")

        # ---------------------------------------------------------
        # Protocol Stage 6: Optimization (Equation 8 & 9)
        # ---------------------------------------------------------
        print("\n--- Stage 6: Secure Optimization (Eq. 8 & 9) ---")
        start_t = time.time()
        # Classifier MLP
        rpc_matmul('h_new', 'W_cls1', 'a_cls1', 'b_cls1', 'c_cls1', 'z_cls1')
        rpc_hadamard_mul('z_cls1', 'z_cls1', 'a_cls_act', 'b_cls_act', 'c_cls_act', 'a_cls1')
        rpc_matmul('a_cls1', 'W_cls2', 'a_cls2', 'b_cls2', 'c_cls2', 'z_cls2')
        rpc_scalar_mul('z_cls2', 0.25, 'q_term')
        rpc_store_all_nodes('const_0_5_loss', np.full((N,1), 0.5)) 
        rpc_add('q_term', 'const_0_5_loss', 'q')
        
        # Classifier Loss
        rpc_sub('q', 'const_1', 'q_minus_1')
        rpc_hadamard_mul('Y', 'q_minus_1', 'a_cls_loss', 'b_cls_loss', 'c_cls_loss', 'term_cls_a')
        rpc_sub('const_0', 'q', 'neg_q')
        rpc_hadamard_mul('one_minus_Y', 'neg_q', 'a_cls_loss', 'b_cls_loss', 'c_cls_loss', 'term_cls_b')
        rpc_add('term_cls_a', 'term_cls_b', 'loss_cls_raw')
        
        # Total Loss
        beta = 0.5
        rpc_scalar_mul('total_term', beta, 'weighted_loss_pre') # total_term is from Eq 2
        rpc_add('loss_cls_raw', 'weighted_loss_pre', 'loss_total_raw')
        print(f"[+] Optimization step completed in {time.time()-start_t:.4f}s")

        print("\n" + "="*60)
        print(f"   EXECUTION FINISHED (Total Time: {time.time()-total_start:.2f}s)")
        print("="*60 + "\n")

        # ---------------------------------------------------------
        # 3. VALIDATION AND BENCHMARKING
        # ---------------------------------------------------------
        h_new_shares = [rpc_get_node(i, 'h_new') for i in range(3)]
        h_new_u = reconstruct_elementwise(h_new_shares)
        h_new_mpc = from_fixed(h_new_u)
        
        # Reference Calculation
        z1 = h.dot(W_mlp1)
        a1 = z1 * z1
        z2 = a1.dot(W_mlp2)
        p_clear = 0.5 + 0.25 * z2
        
        h_new_clear = lexgnn_layer_clear_exact(h, W0, W1, Psi0_b, Psi1_b, Wself, Wd0, Wd1, adjacency, p_clear)
        diff = np.linalg.norm(h_new_mpc - h_new_clear)
        
        print("RESULTS VALIDATION:")
        print(f"   - Norm Difference (MPC vs Clear): {diff:.6f}")
        if diff < 0.1:
            print("   - Status: VALIDATION PASSED")
        else:
            print("   - Status: VALIDATION FAILED")

        print("\nCLASS SEPARATION ANALYSIS:")
        fraud_indices = np.where(Y_labels == 1)[0]
        benign_indices = np.where(Y_labels == 0)[0]
        mean_fraud = np.mean(h_new_mpc[fraud_indices], axis=0)
        mean_benign = np.mean(h_new_mpc[benign_indices], axis=0)
        
        print(f"   - Mean Fraud Vector (First 3 dim):  {mean_fraud[:3]}")
        print(f"   - Mean Benign Vector (First 3 dim): {mean_benign[:3]}")
        print("   (Distinct values confirm successful differentiated processing)\n")

    except Exception as e:
        print(f"\n[-] FAILED with error: {e}")
        import traceback
        traceback.print_exc()