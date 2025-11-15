
def reconstruct_elementwise(shares):
    # elementwise robust reconstruction: sum shares modulo 2**64, return uint64 numpy array
    shape = shares[0].shape
    out = np.zeros(shape, dtype=np.uint64)
    it = np.nditer(out, flags=['multi_index'], op_flags=['readwrite'])
    for _ in it:
        idx = it.multi_index
        s = (int(shares[0][idx]) + int(shares[1][idx]) + int(shares[2][idx])) & ((1 << 64) - 1)
        out[idx] = np.uint64(s)
    return out

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


# debug coordinator: prints per-node a/b/c shares, DE_shares and raw Z patterns
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
            out[idx] = np.int64(val - (1 << 64)) if val >= (1 << 63) else np.int64(val)
        return out
    else:
        return u.view(np.int64)

# fixed-point constants (must match worker)
FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

# small test matrices
M, K, N = 3, 2, 3
A = np.arange(1, 1+M*K).reshape(M,K).astype(np.int64)
B = np.arange(1, 1+K*N).reshape(K,N).astype(np.int64)

# fixedpoint views
A_fixed = (A * SCALE).astype(np.int64)
B_fixed = (B * SCALE).astype(np.int64)
A_u = A_fixed.view(np.uint64)
B_u = B_fixed.view(np.uint64)

# share and send A,B
A_sh = share_uint64(A_u)
B_sh = share_uint64(B_u)
for nid in range(3):
    r = send(nid, {'type':'STORE','key':'A','share':A_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"store A failed on node{nid}: {r}")
    r = send(nid, {'type':'STORE','key':'B','share':B_sh[nid]})
    if not r.get('ok'): raise RuntimeError(f"store B failed on node{nid}: {r}")

# triple generation and distribution
rand_bound = 3
a = np.random.randint(-rand_bound, rand_bound, size=(M,K)).astype(np.int64)
b = np.random.randint(-rand_bound, rand_bound, size=(K,N)).astype(np.int64)
a_fixed = (a * SCALE).astype(np.int64)
b_fixed = (b * SCALE).astype(np.int64)
prod_ab = a_fixed.dot(b_fixed).astype(np.int64)
add = (1 << (FRAC_BITS - 1))
c_fixed = np.right_shift(prod_ab + add, FRAC_BITS).astype(np.int64)

a_u = a_fixed.view(np.uint64); b_u = b_fixed.view(np.uint64); c_u = c_fixed.view(np.uint64)
a_sh = share_uint64(a_u); b_sh = share_uint64(b_u); c_sh = share_uint64(c_u)
for nid in range(3):
    for key,data in [('a',a_sh[nid]),('b',b_sh[nid]),('c',c_sh[nid])]:
        r = send(nid, {'type':'STORE','key':key,'share':data})
        if not r.get('ok'): raise RuntimeError(f"store {key} failed on node{nid}: {r}")

# print per-node stored meta + the raw shares for a,b,c
print("=== per-node metadata and raw shares (a,b,c) ===")
for nid in range(3):
    meta = send(nid, {'type':'DEBUG_KEYS'})
    print(f"node{nid} keys/meta: {meta.get('keys_meta')}")
    for key in ['a','b','c']:
        g = send(nid, {'type':'GET','key':key})
        print(f" node{nid} raw {key} uint64:\n", np.array(g['share'], dtype=np.uint64))

# compute online d/e
d_parts=[]; e_parts=[]
for nid in range(3):
    r = send(nid, {'type':'MAT_COMPUTE_D_E','X_key':'A','Y_key':'B','a_key':'a','b_key':'b'})
    if not r.get('ok'): raise RuntimeError(f"MAT_COMPUTE_D_E failed node{nid}: {r}")
    d_parts.append(np.array(r['d_i'], dtype=np.uint64))
    e_parts.append(np.array(r['e_i'], dtype=np.uint64))
D_u = reconstruct_elementwise(d_parts); E_u = reconstruct_elementwise(e_parts)
D = uint64_to_signed_int64(D_u); E = uint64_to_signed_int64(E_u)

print("Public D (fixed-point int view):\n", D)
print("Public E (fixed-point int view):\n", E)

# compute DE (public) and share it
DE_prod = D.dot(E).astype(np.int64)
DE = np.right_shift(DE_prod + (1 << (FRAC_BITS - 1)), FRAC_BITS).astype(np.int64)
DE_u = DE.view(np.uint64)
print("Public DE (fixed-point int view):\n", DE)
print("Public DE uint64 patterns:\n", DE_u)

DE_sh = share_uint64(DE_u)
# print per-node DE_share
for nid in range(3):
    print(f"node{nid} DE_share uint64:\n", np.array(DE_sh[nid], dtype=np.uint64))
    # also optionally store DE_share on node for inspection
    r = send(nid, {'type':'STORE','key':f'DE_share_{nid}','share': np.array(DE_sh[nid], dtype=np.uint64)})
    if not r.get('ok'): raise RuntimeError(f"store DE_share_{nid} failed on node{nid}: {r}")

# apply online update
for nid in range(3):
    msg = {'type':'APPLY_BACTH_DE_FIXED','D':D.tolist(),'E':E.tolist(),'DE_share':np.array(DE_sh[nid], dtype=np.uint64),
           'c_key':'c','a_key':'a','b_key':'b','store_as':'Z'}
    r = send(nid, msg)
    if not r.get('ok'):
        dbg = send(nid, {'type':'DEBUG_KEYS'})
        raise RuntimeError(f"APPLY failed node{nid}: {r}; keys: {dbg}")

time.sleep(0.05)
# collect raw Z shares
Z_parts=[]
for nid in range(3):
    r = send(nid, {'type':'GET','key':'Z'})
    if not r.get('ok'): raise RuntimeError(f"GET Z failed node{nid}: {r}")
    print(f"node{nid} raw Z uint64:\n", np.array(r['share'], dtype=np.uint64))
    Z_parts.append(np.array(r['share'], dtype=np.uint64))

# reconstruct elementwise
Z_u = reconstruct_elementwise(Z_parts)
print("Reconstructed raw Z uint64:\n", Z_u)
Z_fixed = uint64_to_signed_int64(Z_u).astype(np.int64)
Z_clear = np.round(Z_fixed.astype(np.float64) / SCALE).astype(np.int64)
print("Reconstructed Z (rounded clear):\n", Z_clear)
print("Expected A·B:\n", A.dot(B))
