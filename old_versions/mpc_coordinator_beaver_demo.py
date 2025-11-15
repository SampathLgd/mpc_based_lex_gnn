# mpc_coordinator_beaver_demo.py (final corrected)
import socket, pickle, struct, time, numpy as np

HOST = 'localhost'
PORT_BASE = 9000

def send(node_id, msg, timeout=5.0, retry=5):
    port = PORT_BASE + node_id
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
            time.sleep(0.15)
    raise RuntimeError(f"Failed to send to node {node_id}")

def share_uint64(x_uint):
    s0 = np.random.randint(0, 2**63, size=x_uint.shape, dtype=np.uint64)
    s1 = np.random.randint(0, 2**63, size=x_uint.shape, dtype=np.uint64)
    s2 = (x_uint - s0 - s1).astype(np.uint64)
    return [s0, s1, s2]

# small demo: A(2x2) * B(2x2)
A = np.array([[1,2],[3,4]], dtype=np.int64)
B = np.array([[5,6],[7,8]], dtype=np.int64)
Au = A.view(np.uint64)
Bu = B.view(np.uint64)

# 1) share A and B among three nodes
A_shares = share_uint64(Au)
B_shares = share_uint64(Bu)
for nid in range(3):
    send(nid, {'type':'STORE_SHARES', 'key':'A', 'share': A_shares[nid]})
    send(nid, {'type':'STORE_SHARES', 'key':'B', 'share': B_shares[nid]})

# 2) generate triples for scalar products (naive)
def gen_triples_for_shape(M,K,N):
    triples = []
    rand_bound = 10
    for _ in range(M*K*N):
        a = np.random.randint(-rand_bound, rand_bound, size=(1,)).astype(np.int64)
        b = np.random.randint(-rand_bound, rand_bound, size=(1,)).astype(np.int64)
        c = (a*b).astype(np.int64)
        a_u = a.view(np.uint64); b_u = b.view(np.uint64); c_u = c.view(np.uint64)
        a_sh = share_uint64(a_u); b_sh = share_uint64(b_u); c_sh = share_uint64(c_u)
        triples.append((a_sh, b_sh, c_sh, a_u, b_u, c_u))
    return triples

M,K = A.shape
K,N = B.shape
triples = gen_triples_for_shape(M,K,N)

# distribute triple shares
for idx, (a_sh,b_sh,c_sh,_,_,_) in enumerate(triples):
    for nid in range(3):
        send(nid, {'type':'STORE_SHARES', 'key': f'a_{idx}', 'share': a_sh[nid]})
        send(nid, {'type':'STORE_SHARES', 'key': f'b_{idx}', 'share': b_sh[nid]})
        send(nid, {'type':'STORE_SHARES', 'key': f'c_{idx}', 'share': c_sh[nid]})

# 3) naive Beaver elementwise matmul orchestrated by coordinator
Z_shares = [ np.zeros((M,N), dtype=np.uint64) for _ in range(3) ]

triple_idx = 0
for i in range(M):
    for j in range(N):
        sum_sh = [ np.zeros((1,), dtype=np.uint64) for _ in range(3) ]
        for k in range(K):
            d_res = [ None ] * 3
            e_res = [ None ] * 3
            for nid in range(3):
                resp = send(nid, {'type':'COMPUTE_D_E', 'x_key':'A', 'y_key':'B', 'a_key':f'a_{triple_idx}', 'b_key':f'b_{triple_idx}'})
                d_res[nid] = np.array(resp['d_i'], dtype=np.uint64)
                e_res[nid] = np.array(resp['e_i'], dtype=np.uint64)

            d_open = (d_res[0] + d_res[1] + d_res[2]).astype(np.uint64)
            e_open = (e_res[0] + e_res[1] + e_res[2]).astype(np.uint64)
            d_public = d_open.view(np.int64).astype(np.int64)
            e_public = e_open.view(np.int64).astype(np.int64)
            # take scalar values (they are 1-element arrays)
            d_val = float(np.asarray(d_public).reshape(-1)[0])
            e_val = float(np.asarray(e_public).reshape(-1)[0])

            de_public = (d_public * e_public).astype(np.int64)
            de_u = de_public.view(np.uint64)
            de_sh = share_uint64(de_u)

            # instruct nodes to apply and store product under unique key
            for nid in range(3):
                send(nid, {'type':'APPLY_DE_AND_DE_SHARE',
                           'd': d_val, 'e': e_val,
                           'de_share': de_sh[nid],
                           'c_key': f'c_{triple_idx}', 'a_key': f'a_{triple_idx}', 'b_key': f'b_{triple_idx}',
                           'store_as': f'prod_{i}_{k}_{j}_{triple_idx}' })

            # collect product shares from nodes
            prod_sh_parts = []
            for nid in range(3):
                r = send(nid, {'type':'GET_SHARE', 'key': f'prod_{i}_{k}_{j}_{triple_idx}'})
                prod_sh_parts.append(np.array(r['share'], dtype=np.uint64))

            # accumulate into sum_sh robustly
            for p in range(3):
                part = np.asarray(prod_sh_parts[p]).reshape(-1)
                scalar = np.uint64(part[0])
                sum_sh[p] = (sum_sh[p] + np.array([scalar], dtype=np.uint64)).astype(np.uint64)

            triple_idx += 1

        # store sum_sh into Z_shares at [i,j]
        for p in range(3):
            val_arr = np.asarray(sum_sh[p]).reshape(-1)
            scalar_val = val_arr[0]
            Z_shares[p][i,j] = np.uint64(scalar_val)

# reconstruct Z and display
Z_recon_u = (Z_shares[0] + Z_shares[1] + Z_shares[2]).astype(np.uint64)
Z_recon = Z_recon_u.view(np.int64).astype(np.int64)
print("Reconstructed Z (int view):")
print(Z_recon)
print("Expected A·B:")
print(A.dot(B))
