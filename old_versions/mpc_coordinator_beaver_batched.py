# mpc_coordinator_beaver_batched_fixed.py
import socket, pickle, struct, time, numpy as np

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

def share_uint64(x_uint):
    s0 = np.random.randint(0, 2**63, size=x_uint.shape, dtype=np.uint64)
    s1 = np.random.randint(0, 2**63, size=x_uint.shape, dtype=np.uint64)
    s2 = (x_uint - s0 - s1).astype(np.uint64)
    return [s0, s1, s2]

def reconstruct(shares):
    return (shares[0] + shares[1] + shares[2]).astype(np.uint64)

# Fixed-point parameters (must match worker)
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

# Clear example matrices (small integers)
M, K, N = 3, 2, 3
A = np.arange(1, 1+M*K).reshape(M,K).astype(np.int64)
B = np.arange(1, 1+K*N).reshape(K,N).astype(np.int64)

# Convert to fixed-point representation (int64)
A_fixed = (A.astype(np.int64) * SCALE).astype(np.int64)
B_fixed = (B.astype(np.int64) * SCALE).astype(np.int64)

# Share fixed-point bit-patterns among three nodes (uint64 view)
A_u = A_fixed.view(np.uint64)
B_u = B_fixed.view(np.uint64)
A_sh = share_uint64(A_u)
B_sh = share_uint64(B_u)

# send shares to nodes
for nid in range(3):
    r = send(nid, {'type':'STORE', 'key':'A', 'share': A_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node {nid} STORE A failed: {r}")
    r = send(nid, {'type':'STORE', 'key':'B', 'share': B_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node {nid} STORE B failed: {r}")

# Generate matrix triple a (M x K) and b (K x N) in same fixed domain and c = trunc(a·b)
rand_bound = 3
a = np.random.randint(-rand_bound, rand_bound, size=(M,K)).astype(np.int64)
b = np.random.randint(-rand_bound, rand_bound, size=(K,N)).astype(np.int64)
# scale triple into fixed domain
a_fixed = (a * SCALE).astype(np.int64)
b_fixed = (b * SCALE).astype(np.int64)
# compute product a_fixed @ b_fixed -> scale^2, then truncate
prod_ab = a_fixed.dot(b_fixed).astype(np.int64)  # scale^2
add = (1 << (FRAC_BITS - 1))
c_fixed = np.right_shift(prod_ab + add, FRAC_BITS).astype(np.int64)  # truncated to scale

# share triple as uint64 bit patterns
a_u = a_fixed.view(np.uint64); b_u = b_fixed.view(np.uint64); c_u = c_fixed.view(np.uint64)
a_sh = share_uint64(a_u); b_sh = share_uint64(b_u); c_sh = share_uint64(c_u)

for nid in range(3):
    r = send(nid, {'type':'STORE', 'key':'a', 'share': a_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node {nid} STORE a failed: {r}")
    r = send(nid, {'type':'STORE', 'key':'b', 'share': b_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node {nid} STORE b failed: {r}")
    r = send(nid, {'type':'STORE', 'key':'c', 'share': c_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"node {nid} STORE c failed: {r}")

# Beaver online: compute D_i = A_i - a_i, E_i = B_i - b_i
d_parts = []; e_parts = []
for nid in range(3):
    r = send(nid, {'type':'MAT_COMPUTE_D_E', 'X_key':'A', 'Y_key':'B', 'a_key':'a', 'b_key':'b'})
    if not r.get('ok'): raise RuntimeError(f"node {nid} MAT_COMPUTE_D_E failed: {r}")
    d_parts.append(np.array(r['d_i'], dtype=np.uint64))
    e_parts.append(np.array(r['e_i'], dtype=np.uint64))

D_u = reconstruct(d_parts); E_u = reconstruct(e_parts)
# Interpret as signed int64 fixed-point numbers
D = D_u.view(np.int64).astype(np.int64)   # fixed-point (scale)
E = E_u.view(np.int64).astype(np.int64)

# compute public DE = trunc(D @ E) (D,E are fixed-point -> product scale^2)
DE_prod = D.dot(E).astype(np.int64)  # scale^2
DE = np.right_shift(DE_prod + (1 << (FRAC_BITS - 1)), FRAC_BITS).astype(np.int64)  # back to scale
DE_u = DE.view(np.uint64)
DE_shares = share_uint64(DE_u)

# Now broadcast D (as int fixed-point), E and each node's DE_share
for nid in range(3):
    # D and E are int64 numpy arrays in fixed domain; send as python lists of ints (int64 fits in pickle)
    msg = {'type':'APPLY_BACTH_DE_FIXED', 'D': D.tolist(), 'E': E.tolist(), 'DE_share': np.array(DE_shares[nid], dtype=np.uint64),
           'c_key': 'c', 'a_key': 'a', 'b_key': 'b', 'store_as': 'Z'}
    r = send(nid, msg)
    if not r.get('ok'):
        dbg = send(nid, {'type':'DEBUG_KEYS'})
        raise RuntimeError(f"node {nid} APPLY failed: {r}; keys: {dbg}")

time.sleep(0.05)

# collect Z shares
Z_sh = [None]*3
for nid in range(3):
    r = send(nid, {'type':'GET', 'key':'Z'})
    if not r.get('ok'): raise RuntimeError(f"node {nid} GET Z failed: {r}")
    if r.get('share') is None: raise RuntimeError(f"node {nid} returned None for Z; keys: {send(nid, {'type':'DEBUG_KEYS'})}")
    Z_sh[nid] = np.array(r['share'], dtype=np.uint64)

Z_u = reconstruct(Z_sh)
# Convert back to clear integers: Z_u.view(int64) are fixed-point integers; divide by SCALE to get real numbers.
Z_fixed = Z_u.view(np.int64).astype(np.int64)
# Final clear matrix (rounded)
Z_clear = np.round(Z_fixed.astype(np.float64) / SCALE).astype(np.int64)
print("Reconstructed Z (rounded clear int view):")
print(Z_clear)
print("Expected A·B (clear):")
print(A.dot(B))
