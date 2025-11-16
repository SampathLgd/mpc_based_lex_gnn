# mpc_gnn_coordinator_v1.py
# Assembles all MPC primitives to run one GNN layer.
# This version computes attention in the clear, per lex_gnn_mpc_sim.py.

import socket, pickle, struct, time, numpy as np

# ----------
# 1. COPY ALL HELPER FUNCTIONS
# (reconstruct_elementwise, send, rpc_add, rpc_sub, rpc_hadamard_mul,
#  rpc_scalar_mul, rpc_get_node, share_uint64, uint64_to_signed_int64)
# ----------

def reconstruct_elementwise(shares_list):
    if not shares_list or len(shares_list) != 3:
        raise ValueError("Requires a list of 3 shares to reconstruct.")
    s0 = np.asarray(shares_list[0], dtype=np.uint64)
    s1 = np.asarray(shares_list[1], dtype=np.uint64)
    s2 = np.asarray(shares_list[2], dtype=np.uint64)
    return (s0 + s1 + s2).astype(np.uint64)

HOST = 'localhost'
PORT_BASE = 9000

def send(node_id, msg, timeout=5.0, retry=6):
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

def rpc_add(key_A, key_B, store_as):
    print(f"Instructing nodes: [ {store_as} ] = [ {key_A} ] + [ {key_B} ]")
    req = {'type': 'LOCAL_ADD', 'key_A': key_A, 'key_B': key_B, 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} LOCAL_ADD failed: {r}")
    return True

def rpc_sub(key_A, key_B, store_as):
    print(f"Instructing nodes: [ {store_as} ] = [ {key_A} ] - [ {key_B} ]")
    req = {'type': 'LOCAL_SUB', 'key_A': key_A, 'key_B': key_B, 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} LOCAL_SUB failed: {r}")
    return True

# --- This is the MATMUL rpc_... logic from your other coordinator ---
# --- We need to copy it here ---
def rpc_matmul(X_key, Y_key, a_key, b_key, c_key, store_as):
    print(f"Instructing nodes: [ {store_as} ] = [ {X_key} ] @ [ {Y_key} ]")
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

    DE_prod = D.dot(E).astype(np.int64)
    DE = np.right_shift(DE_prod, FRAC_BITS).astype(np.int64)
    DE_u = DE.view(np.uint64)
    DE_sh = share_uint64(DE_u)

    for nid in range(3):
        msg = {'type':'APPLY_BACTH_DE_FIXED', 'D': D.tolist(), 'E': E.tolist(), 'DE_share': np.array(DE_sh[nid], dtype=np.uint64),
               'c_key':c_key,'a_key':a_key,'b_key':b_key,'store_as':store_as}
        r = send(nid, msg)
        if not r.get('ok'):
            dbg = send(nid, {'type':'DEBUG_KEYS'})
            raise RuntimeError(f"node{nid} APPLY failed: {r}; keys: {dbg}")
    time.sleep(0.05)
    return True
# --- End of rpc_matmul ---

def rpc_hadamard_mul(X_key, Y_key, a_key, b_key, c_key, store_as):
    print(f"Instructing nodes: [ {store_as} ] = [ {X_key} ] .* [ {Y_key} ]")
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
    time.sleep(0.05)
    return True

def rpc_scalar_mul(key_in, scalar_float, store_as):
    print(f"Instructing nodes: [ {store_as} ] = {scalar_float} * [ {key_in} ]")
    scalar_fixed = np.round(scalar_float * SCALE).astype(np.int64)
    req = {'type': 'PUBLIC_SCALAR_MUL', 'key_in': key_in,
           'scalar_fixed': scalar_fixed.item(), 'store_as': store_as}
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'): raise Exception(f"node{nid} PUBLIC_SCALAR_MUL failed: {r}")
    time.sleep(0.05)
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
    s2_list = []
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        xi = int(x[idx]); v0 = int(s0[idx]); v1 = int(s1[idx])
        s2_val = (xi - v0 - v1) & ((1 << 64) - 1)
        s2_list.append(s2_val)
    s2 = np.array(s2_list, dtype=np.uint64).reshape(x.shape)
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
    """Convert uint64 array (two's complement) back to float numpy array."""
    return x_uint.view(np.int64).astype(np.float64) / SCALE

def to_fixed(x):
    """Convert float numpy array to fixed-point uint64 (two's complement view)."""
    xi = np.round(x * SCALE).astype(np.int64)
    return xi.view(np.uint64)

# --- NEW HELPER ---
def rpc_store_all_nodes(key, clear_data_float):
    """Shares a clear float array and stores one share on each node."""
    print(f"Sharing and storing [ {key} ]...")
    data_u = to_fixed(clear_data_float)
    shares = share_uint64(data_u)
    for nid in range(3):
        r = send(nid, {'type':'STORE', 'key':key, 'share': shares[nid]})
        if not r.get('ok'): raise RuntimeError(f"node{nid} STORE {key} failed {r}")
    return True

# --- NEW HELPER ---
def gen_and_store_triples(shape, key_prefix, rand_bound=3, op_type='matmul'):
    """Generates and stores triples for matmul or hadamard."""
    print(f"Generating and storing triples for [ {key_prefix} ]...")
    if op_type == 'matmul':
        M, K, N = shape
        a = np.random.randint(-rand_bound, rand_bound, size=(M,K)).astype(np.int64)
        b = np.random.randint(-rand_bound, rand_bound, size=(K,N)).astype(np.int64)
        a_f = (a * SCALE).astype(np.int64)
        b_f = (b * SCALE).astype(np.int64)
        prod_ab = a_f.dot(b_f).astype(np.int64)
        c_f = np.right_shift(prod_ab, FRAC_BITS).astype(np.int64)
    else: # hadamard
        M, N = shape
        a = np.random.randint(-rand_bound, rand_bound, size=(M,N)).astype(np.int64)
        b = np.random.randint(-rand_bound, rand_bound, size=(M,N)).astype(np.int64)
        a_f = (a * SCALE).astype(np.int64)
        b_f = (b * SCALE).astype(np.int64)
        prod_ab = (a_f * b_f).astype(np.int64)
        c_f = np.right_shift(prod_ab, FRAC_BITS).astype(np.int64)

    a_u, b_u, c_u = a_f.view(np.uint64), b_f.view(np.uint64), c_f.view(np.uint64)
    a_sh, b_sh, c_sh = share_uint64(a_u), share_uint64(b_u), share_uint64(c_u)

    for nid in range(3):
        for key,data in [(f'a_{key_prefix}', a_sh[nid]),
                         (f'b_{key_prefix}', b_sh[nid]),
                         (f'c_{key_prefix}', c_sh[nid])]:
            r = send(nid, {'type':'STORE','key':key,'share':data})
            if not r.get('ok'): raise RuntimeError(f"node{nid} STORE {key} failed {r}")
    return True

# --- Cleartext reference function (from lex_gnn_mpc_sim.py) ---
def poly_sigmoid_clear(x):
    return 0.5 + 0.25 * x - (x**3) / 48.0

def lexgnn_layer_clear(h, W0, W1, Wself, Wdest, Psi0, Psi1, adjacency, p_clear):
    N = h.shape[0]
    p_clear = p_clear.reshape(N, 1) # Ensure (N,1)
    
    base = h.dot(W0)
    diff = h.dot(W1 - W0)
    
    # This is the simplified message from your no_recon sim
    p_diff = diff * p_clear
    m = base + p_diff
    
    # Public attention
    agg = np.zeros_like(m)
    for v in range(N):
        neighs = adjacency[v]
        if len(neighs) == 0: continue
        
        scores = []
        for u in neighs:
            # Replicate the no_recon sim's dot-and-square
            sc = np.dot(m[u], m[v])
            scores.append(sc * sc)
        scores = np.array(scores, dtype=np.float64)
        
        ssum = scores.sum()
        if ssum == 0 or scores.size == 0:
            weights = np.ones_like(scores) / max(1, scores.size)
        else:
            weights = scores / ssum
            
        for idx, u in enumerate(neighs):
            agg[v] += weights[idx] * m[u]

    base_d = agg.dot(Wdest)
    wself_h = h.dot(Wself)
    h_new = wself_h + base_d
    return h_new


# ----------
# 2. NEW MAIN BLOCK
# ----------
if __name__ == "__main__":
    print("--- Starting GNN Layer v1 Coordinator ---")
    
    # 1. Setup toy data (from lex_gnn_mpc_no_recon.py)
    N=6; D=4; Dout=3
    np.random.seed(42)
    h = np.random.randn(N,D)
    W0 = np.random.randn(D,Dout)*0.5
    W1 = W0 + np.random.randn(D,Dout)*0.1
    Wself = np.random.randn(D,Dout)*0.2
    Wdest = np.random.randn(Dout,Dout)*0.3
    # Note: Skipping Psi0/Psi1 for this test, matches sim
    adjacency = {i:[(i-1)%N,(i+1)%N] for i in range(N)}
    adjacency[0].append(2); adjacency[3].append(1)

    # Simulated p_clear (from lex_gnn_mpc_no_recon.py)
    z = h.dot(np.random.randn(h.shape[1],1)*0.1)
    p_clear = 0.5 + 0.25 * z.flatten() - (z.flatten()**3) / 48.0
    p_clear_b = np.tile(p_clear.reshape(N,1), (1, Dout)) # Broadcast p to (N, Dout)

    print(f"Setup: N={N}, D={D}, Dout={Dout}")

    try:
        # 2. Share all inputs
        rpc_store_all_nodes('h', h)
        rpc_store_all_nodes('W0', W0)
        rpc_store_all_nodes('W1', W1)
        rpc_store_all_nodes('Wself', Wself)
        rpc_store_all_nodes('Wdest', Wdest)
        rpc_store_all_nodes('p', p_clear_b) # Store the (N, Dout) broadcasted version
        
        # 3. Generate and store all triples needed
        gen_and_store_triples((N,D,Dout), 'base', op_type='matmul')  # h @ W0
        gen_and_store_triples((N,D,Dout), 'diff', op_type='matmul')  # h @ Wdiff
        gen_and_store_triples((N,Dout), 'pdiff', op_type='hadamard') # diff .* p
        gen_and_store_triples((N,Dout,Dout), 'based', op_type='matmul') # agg @ Wdest
        gen_and_store_triples((N,D,Dout), 'wselfh', op_type='matmul') # h @ Wself
        
        # 4. --- Run Secure GNN Layer ---
        
        # [base] = [h] @ [W0]
        rpc_matmul('h', 'W0', 'a_base', 'b_base', 'c_base', 'base')
        
        # [Wdiff] = [W1] - [W0]
        rpc_sub('W1', 'W0', 'Wdiff')
        
        # [diff] = [h] @ [Wdiff]
        rpc_matmul('h', 'Wdiff', 'a_diff', 'b_diff', 'c_diff', 'diff')
        
        # [p_diff] = [diff] .* [p]
        rpc_hadamard_mul('diff', 'p', 'a_pdiff', 'b_pdiff', 'c_pdiff', 'p_diff')
        
        # [m] = [base] + [p_diff]
        rpc_add('base', 'p_diff', 'm')

        print("✅ Secure message computation complete.")

        # 5. --- Public Attention (sim.py logic) ---
        print("Reconstructing messages [m] for public attention...")
        m_shares = [rpc_get_node(i, 'm') for i in range(3)]
        m_u = reconstruct_elementwise(m_shares)
        m_clear = from_fixed(m_u)
        
        print("Running public attention...")
        agg_clear = np.zeros_like(m_clear)
        for v in range(N):
            neighs = adjacency[v]
            if len(neighs) == 0: continue
            
            scores = []
            for u in neighs:
                sc = np.dot(m_clear[u], m_clear[v])
                scores.append(sc * sc)
            scores = np.array(scores, dtype=np.float64)

            ssum = scores.sum()
            if ssum == 0 or scores.size == 0:
                weights = np.ones_like(scores) / max(1, scores.size)
            else:
                weights = scores / ssum
            
            for idx, u in enumerate(neighs):
                agg_clear[v] += weights[idx] * m_clear[u]

        print("✅ Public aggregation complete.")

        # 6. --- Secure Final Projection ---
        
        # Re-share agg_clear -> [agg]
        rpc_store_all_nodes('agg', agg_clear)
        
        # [base_d] = [agg] @ [Wdest]
        rpc_matmul('agg', 'Wdest', 'a_based', 'b_based', 'c_based', 'base_d')
        
        # [wself_h] = [h] @ [Wself]
        rpc_matmul('h', 'Wself', 'a_wselfh', 'b_wselfh', 'c_wselfh', 'wself_h')
        
        # [h_new] = [wself_h] + [base_d]
        rpc_add('wself_h', 'base_d', 'h_new')
        
        print("✅ Secure final projection complete.")

        # 7. --- Verification ---
        print("Reconstructing final [h_new]...")
        h_new_shares = [rpc_get_node(i, 'h_new') for i in range(3)]
        h_new_u = reconstruct_elementwise(h_new_shares)
        h_new_mpc = from_fixed(h_new_u)
        
        print("Running cleartext reference computation...")
        h_new_clear = lexgnn_layer_clear(h, W0, W1, Wself, Wdest, None, None, adjacency, p_clear)
        
        print("\n--- COMPARISON ---")
        print("MPC h_new (norms):", np.linalg.norm(h_new_mpc, axis=1))
        print("Clear h_new (norms):", np.linalg.norm(h_new_clear, axis=1))
        
        diff = np.linalg.norm(h_new_mpc - h_new_clear)
        print(f"Difference (norm): {diff}")
        
        if diff < 1e-2:
            print("\n✅ GNN Layer v1 PASSED!")
        else:
            print(f"\n❌ GNN Layer v1 FAILED! High difference: {diff}")

    except Exception as e:
        print(f"\n❌ GNN Layer v1 FAILED with error: {e}")
        import traceback
        traceback.print_exc()