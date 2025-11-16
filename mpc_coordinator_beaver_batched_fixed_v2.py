# coordinator v2: checks per-node shapes before running Beaver online
import socket, pickle, struct, time, numpy as np

def reconstruct_elementwise(shares_list):
    """
    Reconstructs a secret from a list of 3 uint64 numpy array shares.
    shares_list: [share_node_0, share_node_1, share_node_2]
    """
    if not shares_list or len(shares_list) != 3:
        raise ValueError("Requires a list of 3 shares to reconstruct.")
    
    # Element-wise addition (modulo 2^64 is handled by uint64)
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


# ... after send() function ...

def rpc_add(key_A, key_B, store_as):
    """Instructs all nodes to add shares [A] + [B] and store as [store_as]"""
    print(f"Instructing nodes: [ {store_as} ] = [ {key_A} ] + [ {key_B} ]")
    req = {
        'type': 'LOCAL_ADD',
        'key_A': key_A,
        'key_B': key_B,
        'store_as': store_as
    }
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'):
            raise Exception(f"node{nid} LOCAL_ADD failed: {r}")
    return True

def rpc_sub(key_A, key_B, store_as):
    """Instructs all nodes to subtract shares [A] - [B] and store as [store_as]"""
    print(f"Instructing nodes: [ {store_as} ] = [ {key_A} ] - [ {key_B} ]")
    req = {
        'type': 'LOCAL_SUB',
        'key_A': key_A,
        'key_B': key_B,
        'store_as': store_as
    }
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'):
            raise Exception(f"node{nid} LOCAL_SUB failed: {r}")
    return True

def rpc_hadamard_mul(X_key, Y_key, a_key, b_key, c_key, store_as):
    """
    Performs secure element-wise multiplication [Z] = [X] .* [Y]
    using Beaver triples [a], [b], [c] where c = a .* b.
    """
    print(f"Instructing nodes: [ {store_as} ] = [ {X_key} ] .* [ {Y_key} ]")
    
    # 1. Compute D_i = [X_i] - [a_i] and E_i = [Y_i] - [b_i] on each node
    d_parts=[]; e_parts=[]
    for nid in range(3):
        # We can reuse the MAT_COMPUTE_D_E RPC since it's just element-wise subtraction
        r = send(nid, {'type':'MAT_COMPUTE_D_E', 'X_key': X_key,'Y_key': Y_key,'a_key': a_key,'b_key': b_key})
        if not r.get('ok'): raise RuntimeError(f"node{nid} MAT_COMPUTE_D_E failed: {r}")
        d_parts.append(np.array(r['d_i'], dtype=np.uint64))
        e_parts.append(np.array(r['e_i'], dtype=np.uint64))

    # 2. Reconstruct D, E
    D_u = reconstruct_elementwise(d_parts)
    E_u = reconstruct_elementwise(e_parts)
    D = uint64_to_signed_int64(D_u)
    E = uint64_to_signed_int64(E_u)

    # 3. Compute public DE = D .* E (element-wise) and truncate
    # This is scale^2, needs truncation
    DE_prod = (D.astype(np.int64) * E.astype(np.int64)).astype(np.int64)
    # --- MODIFIED ---
    DE = np.right_shift(DE_prod, FRAC_BITS).astype(np.int64)
    DE_u = DE.view(np.uint64)
    DE_sh = share_uint64(DE_u) # Share the truncated result

    # 4. Send D, E, DE_sh to nodes and apply
    for nid in range(3):
        msg = {'type':'APPLY_HADAMARD_DE_FIXED', 
               'D': D.tolist(), 
               'E': E.tolist(), 
               'DE_share': np.array(DE_sh[nid], dtype=np.uint64),
               'a_key': a_key,
               'b_key': b_key,
               'c_key': c_key,
               'store_as': store_as}
        r = send(nid, msg)
        if not r.get('ok'):
            dbg = send(nid, {'type':'DEBUG_KEYS'})
            raise RuntimeError(f"node{nid} APPLY_HADAMARD_DE_FIXED failed: {r}; keys: {dbg}")
    
    time.sleep(0.05) # Give nodes time to compute
    return True

def rpc_scalar_mul(key_in, scalar_float, store_as):
    """
    Performs secure public scalar multiplication [Y] = y * [X]
    """
    print(f"Instructing nodes: [ {store_as} ] = {scalar_float} * [ {key_in} ]")
    
    # Convert the public float to fixed-point
    scalar_fixed = np.round(scalar_float * SCALE).astype(np.int64)
    
    req = {
        'type': 'PUBLIC_SCALAR_MUL',
        'key_in': key_in,
        'scalar_fixed': scalar_fixed.item(), # Send as a standard Python int
        'store_as': store_as
    }
    
    for nid in range(3):
        r = send(nid, req)
        if not r.get('ok'):
            raise Exception(f"node{nid} PUBLIC_SCALAR_MUL failed: {r}")
    
    time.sleep(0.05) # Give nodes time to compute
    return True

def rpc_get_node(nid, key):
    """Gets a share from a single node"""
    r = send(nid, {'type':'GET', 'key':key})
    if not r.get('ok') or r.get('share') is None:
        dbg = send(nid, {'type':'DEBUG_KEYS'})
        raise Exception(f"node{nid} GET {key} failed: {r}; keys: {dbg}")
    return np.array(r['share'], dtype=np.uint64)

def share_uint64(x_uint):
    # Use small random splits for simulation to avoid huge intermediate products.
    # This keeps shares random but bounds magnitude to avoid int64 overflow in the sim.
    x = np.asarray(x_uint, dtype=np.uint64)
    rand_share_bound = 1 << 20   # ~1 million — adjust if you need larger dynamic range
    s0 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s1 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    # compute s2 elementwise with Python ints to avoid NumPy width/wrap issues
    s2_list = []
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        xi = int(x[idx])
        v0 = int(s0[idx])
        v1 = int(s1[idx])
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
            if val >= (1 << 63):
                out[idx] = np.int64(val - (1 << 64))
            else:
                out[idx] = np.int64(val)
        return out
    else:
        return u.view(np.int64)

# fixed point params
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

# example clear matrices
M,K,N = 3,2,3
A = np.arange(1,1+M*K).reshape(M,K).astype(np.int64)
B = np.arange(1,1+K*N).reshape(K,N).astype(np.int64)

# convert to fixed domain
A_fixed = (A * SCALE).astype(np.int64); B_fixed = (B * SCALE).astype(np.int64)
A_u = A_fixed.view(np.uint64); B_u = B_fixed.view(np.uint64)
A_sh = share_uint64(A_u); B_sh = share_uint64(B_u)

# store shares to nodes
for nid in range(3):
    r = send(nid, {'type':'STORE', 'key':'A', 'share': A_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node{nid} STORE A failed {r}")
    r = send(nid, {'type':'STORE', 'key':'B', 'share': B_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node{nid} STORE B failed {r}")

# triple generation and distribution
rand_bound = 3
a = np.random.randint(-rand_bound, rand_bound, size=(M,K)).astype(np.int64)
b = np.random.randint(-rand_bound, rand_bound, size=(K,N)).astype(np.int64)
a_fixed = (a * SCALE).astype(np.int64); b_fixed = (b * SCALE).astype(np.int64)
prod_ab = a_fixed.dot(b_fixed).astype(np.int64)
# --- MODIFIED ---
c_fixed = np.right_shift(prod_ab, FRAC_BITS).astype(np.int64)
a_u = a_fixed.view(np.uint64); b_u = b_fixed.view(np.uint64); c_u = c_fixed.view(np.uint64)
a_sh = share_uint64(a_u); b_sh = share_uint64(b_u); c_sh = share_uint64(c_u)

for nid in range(3):
    for key,data in [('a',a_sh[nid]),('b',b_sh[nid]),('c',c_sh[nid])]:
        r = send(nid, {'type':'STORE','key':key,'share':data})
        if not r.get('ok'): raise RuntimeError(f"node{nid} STORE {key} failed {r}")

# ====== SHAPE CHECK PHASE (new) =======
print("=== per-node stored metadata BEFORE online phase ===")
per_node_meta = {}
keys_to_check = ['A','B','a','b','c']
for nid in range(3):
    meta = {}
    for k in keys_to_check:
        r = send(nid, {'type':'GET_SHAPE','key':k})
        if not r.get('ok'):
            raise RuntimeError(f"node{nid} GET_SHAPE {k} failed: {r}")
        meta[k] = r.get('meta')
    per_node_meta[nid] = meta
    print(f"node{nid}:", meta)

# verify all nodes report identical shapes for each key
for k in keys_to_check:
    shapes = [ per_node_meta[nid][k]['shape'] if per_node_meta[nid][k] else None for nid in range(3) ]
    if not all(s == shapes[0] for s in shapes):
        raise RuntimeError(f"Shape mismatch for key {k}: per-node shapes = {shapes}. Aborting. Inspect node metadata above.")

print("shapes ok — proceeding to Beaver online phase")

# Beaver online: compute D_i and E_i
d_parts=[]; e_parts=[]
for nid in range(3):
    r = send(nid, {'type':'MAT_COMPUTE_D_E', 'X_key':'A','Y_key':'B','a_key':'a','b_key':'b'})
    if not r.get('ok'): raise RuntimeError(f"node{nid} MAT_COMPUTE_D_E failed: {r}")
    d_parts.append(np.array(r['d_i'], dtype=np.uint64))
    e_parts.append(np.array(r['e_i'], dtype=np.uint64))

# reconstruct D,E elementwise and convert to signed int64 fixed
D_u = reconstruct_elementwise(d_parts); E_u = reconstruct_elementwise(e_parts)
D = uint64_to_signed_int64(D_u); E = uint64_to_signed_int64(E_u)

# compute DE public (trunc(D@E))
DE_prod = D.dot(E).astype(np.int64)
# --- MODIFIED ---
DE = np.right_shift(DE_prod, FRAC_BITS).astype(np.int64)
DE_u = DE.view(np.uint64)
DE_sh = share_uint64(DE_u)

# send D/E/DE_share and instruct nodes to compute Z
for nid in range(3):
    msg = {'type':'APPLY_BACTH_DE_FIXED', 'D': D.tolist(), 'E': E.tolist(), 'DE_share': np.array(DE_sh[nid], dtype=np.uint64),
           'c_key':'c','a_key':'a','b_key':'b','store_as':'Z'}
    r = send(nid, msg)
    if not r.get('ok'):
        dbg = send(nid, {'type':'DEBUG_KEYS'})
        raise RuntimeError(f"node{nid} APPLY failed: {r}; keys: {dbg}")

time.sleep(0.05)

# collect Z shares and reconstruct
Z_parts = []
for nid in range(3):
    r = send(nid, {'type':'GET','key':'Z'})
    if not r.get('ok'): raise RuntimeError(f"node{nid} GET Z failed: {r}")
    if r.get('share') is None: raise RuntimeError(f"node{nid} returned None for Z; debug: {send(nid, {'type':'DEBUG_KEYS'})}")
    Z_parts.append(np.array(r['share'], dtype=np.uint64))

Z_u = reconstruct_elementwise(Z_parts)
# convert to signed int64 fixed
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

Z_fixed = uint64_to_signed_int64(Z_u).astype(np.int64)
Z_clear = np.round(Z_fixed.astype(np.float64) / SCALE).astype(np.int64)

print("Reconstructed Z (rounded clear int view):")
print(Z_clear)
print("Expected A·B (clear):")
print(A.dot(B))

# --- ADD THIS TEST BLOCK FOR LOCAL_ADD ---
print("\n--- Testing LOCAL_ADD ---")
# Test: compute [Z] + [Z] and store as [Z_plus_Z]
try:
    rpc_add('Z', 'Z', 'Z_plus_Z')
    
    # Get the shares of the new result
    Z_plus_Z_shares = [ rpc_get_node(i, 'Z_plus_Z') for i in range(3) ]
    Z_plus_Z_recon_u = reconstruct_elementwise(Z_plus_Z_shares)
    
    # Compare to cleartext
    # We already have Z_u from the matmul test
    Z_plus_Z_expected_u = (Z_u.astype(np.uint64) + Z_u.astype(np.uint64)).astype(np.uint64)
    
    print("Reconstructed Z+Z (uint64 view):")
    print(Z_plus_Z_recon_u)
    print("Expected Z+Z (uint64 view):")
    print(Z_plus_Z_expected_u)
    
    if np.array_equal(Z_plus_Z_recon_u, Z_plus_Z_expected_u):
        print("✅ LOCAL_ADD test PASSED!")
    else:
        print("❌ LOCAL_ADD test FAILED!")
        
except Exception as e:
    print(f"❌ LOCAL_ADD test FAILED with error: {e}")
# --- END OF TEST BLOCK ---

print("\n--- Testing LOCAL_SUB ---")
# Test: compute [Z_plus_Z] - [Z] and store as [Z_sub_test]
try:
    rpc_sub('Z_plus_Z', 'Z', 'Z_sub_test')
    
    # Get the shares of the new result
    Z_sub_shares = [ rpc_get_node(i, 'Z_sub_test') for i in range(3) ]
    Z_sub_recon_u = reconstruct_elementwise(Z_sub_shares)
    
    # Compare to cleartext
    # The expected result is just Z_u
    Z_sub_expected_u = Z_u.astype(np.uint64)
    
    print("Reconstructed (Z+Z)-Z (uint64 view):")
    print(Z_sub_recon_u)
    print("Expected Z (uint64 view):")
    print(Z_sub_expected_u)
    
    if np.array_equal(Z_sub_recon_u, Z_sub_expected_u):
        print("✅ LOCAL_SUB test PASSED!")
    else:
        print("❌ LOCAL_SUB test FAILED!")
        
except Exception as e:
    print(f"❌ LOCAL_SUB test FAILED with error: {e}")
    
    
print("\n--- Testing HADAMARD_MUL ---")
try:
    # 1. Generate new triples a, b, c where c = (a .* b) >> FRAC_BITS
    # We will test [Z_squared] = [Z] .* [Z]
    # Z has shape (M, N)
    a_h = np.random.randint(-rand_bound, rand_bound, size=(M,N)).astype(np.int64)
    b_h = np.random.randint(-rand_bound, rand_bound, size=(M,N)).astype(np.int64)
    
    a_h_fixed = (a_h * SCALE).astype(np.int64)
    b_h_fixed = (b_h * SCALE).astype(np.int64)
    
    # c = a .* b (element-wise), then truncate
    prod_ab_h = (a_h_fixed.astype(np.int64) * b_h_fixed.astype(np.int64)).astype(np.int64)
    # --- MODIFIED ---
    c_h_fixed = np.right_shift(prod_ab_h, FRAC_BITS).astype(np.int64)
    
    a_h_u = a_h_fixed.view(np.uint64)
    b_h_u = b_h_fixed.view(np.uint64)
    c_h_u = c_h_fixed.view(np.uint64)
    
    a_h_sh = share_uint64(a_h_u)
    b_h_sh = share_uint64(b_h_u)
    c_h_sh = share_uint64(c_h_u)

    # 2. Store these triples on the nodes
    print("Storing Hadamard triples (a_h, b_h, c_h)...")
    for nid in range(3):
        for key,data in [('a_h', a_h_sh[nid]), ('b_h', b_h_sh[nid]), ('c_h', c_h_sh[nid])]:
            r = send(nid, {'type':'STORE','key':key,'share':data})
            if not r.get('ok'): raise RuntimeError(f"node{nid} STORE {key} failed {r}")

    # 3. Run the Hadamard RPC
    rpc_hadamard_mul('Z', 'Z', 'a_h', 'b_h', 'c_h', 'Z_squared')

    # 4. Get the shares and reconstruct
    Z_squared_shares = [ rpc_get_node(i, 'Z_squared') for i in range(3) ]
    Z_squared_recon_u = reconstruct_elementwise(Z_squared_shares)
    
    # 5. Compare to cleartext
    # Z_u is from the matmul test
    Z_fixed = uint64_to_signed_int64(Z_u)
    # Z_squared_expected = (Z_fixed * Z_fixed) >> FRAC_BITS
    Z_squared_prod = (Z_fixed.astype(np.int64) * Z_fixed.astype(np.int64)).astype(np.int64)
    # --- MODIFIED ---
    Z_squared_expected_fixed = np.right_shift(Z_squared_prod, FRAC_BITS).astype(np.int64)
    Z_squared_expected_u = Z_squared_expected_fixed.view(np.uint64)
    
    print("Reconstructed Z*Z (uint64 view):")
    print(Z_squared_recon_u)
    print("Expected Z*Z (uint64 view):")
    print(Z_squared_expected_u)
    
    # Use a tolerance for fixed-point comparisons
    diff = np.abs(Z_squared_recon_u.view(np.int64) - Z_squared_expected_u.view(np.int64))
    if np.all(diff <= 1): # Allow off-by-one errors from intermediate shifts
        print("✅ HADAMARD_MUL test PASSED! (with tolerance)")
    else:
        print("❌ HADAMARD_MUL test FAILED!")
        
except Exception as e:
    print(f"❌ HADAMARD_MUL test FAILED with error: {e}")

print("\n--- Testing PUBLIC_SCALAR_MUL ---")
try:
    # 1. Test: compute [Z_times_3] = 3.5 * [Z]
    public_scalar = 3.5
    rpc_scalar_mul('Z', public_scalar, 'Z_times_3point5')

    # 2. Get the shares and reconstruct
    Z_times_3_shares = [ rpc_get_node(i, 'Z_times_3point5') for i in range(3) ]
    Z_times_3_recon_u = reconstruct_elementwise(Z_times_3_shares)
    
    # 3. Compare to cleartext
    Z_fixed = uint64_to_signed_int64(Z_u)
    scalar_fixed = np.round(public_scalar * SCALE).astype(np.int64)
    
    # Expected result: trunc( (Z_fixed * 3.5_fixed) )
    prod = (Z_fixed.astype(np.int64) * scalar_fixed).astype(np.int64)
    # --- MODIFIED ---
    expected_fixed = np.right_shift(prod, FRAC_BITS).astype(np.int64)
    expected_u = expected_fixed.view(np.uint64)
    
    print("Reconstructed 3.5*Z (uint64 view):")
    print(Z_times_3_recon_u)
    print("Expected 3.5*Z (uint64 view):")
    print(expected_u)
    
    if np.array_equal(Z_times_3_recon_u, expected_u):
        print("✅ PUBLIC_SCALAR_MUL test PASSED!")
    else:
        # Check for off-by-one
        diff = np.abs(Z_times_3_recon_u.view(np.int64) - expected_u.view(np.int64))
        if np.all(diff <= 1):
            print("✅ PUBLIC_SCALAR_MUL test PASSED! (with tolerance)")
        else:
            print("❌ PUBLIC_SCALAR_MUL test FAILED!")
            print("Diff:", diff)
        
except Exception as e:
    print(f"❌ PUBLIC_SCALAR_MUL test FAILED with error: {e}")
# --- END OF TEST BLOCK ---