# mpc_coordinator_beaver_batched_debug.py
# Debug version of batched Beaver matmul coordinator
import socket, pickle, struct, time, numpy as np

HOST = 'localhost'
PORT_BASE = 9000

def send(node_id, msg, timeout=5.0, retry=5):
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

M, K, N = 3, 2, 3
A = np.arange(1, 1+M*K).reshape(M,K).astype(np.int64)
B = np.arange(1, 1+K*N).reshape(K,N).astype(np.int64)

Au, Bu = A.view(np.uint64), B.view(np.uint64)

A_sh, B_sh = share_uint64(Au), share_uint64(Bu)
for nid in range(3):
    for key, data in [('A',A_sh[nid]), ('B',B_sh[nid])]:
        r = send(nid, {'type':'STORE','key':key,'share':data})
        if not r.get('ok'): raise RuntimeError(f"Node {nid} STORE {key} failed: {r}")

rand_bound = 5
a = np.random.randint(-rand_bound, rand_bound, size=(M,K)).astype(np.int64)
b = np.random.randint(-rand_bound, rand_bound, size=(K,N)).astype(np.int64)
c = a.dot(b).astype(np.int64)
a_u,b_u,c_u = a.view(np.uint64), b.view(np.uint64), c.view(np.uint64)
a_sh,b_sh,c_sh = share_uint64(a_u), share_uint64(b_u), share_uint64(c_u)
for nid in range(3):
    for key,data in [('a',a_sh[nid]),('b',b_sh[nid]),('c',c_sh[nid])]:
        r = send(nid,{'type':'STORE','key':key,'share':data})
        if not r.get('ok'): raise RuntimeError(f"Node {nid} STORE {key} failed: {r}")

d_parts,e_parts=[],[]
for nid in range(3):
    r=send(nid,{'type':'MAT_COMPUTE_D_E','X_key':'A','Y_key':'B','a_key':'a','b_key':'b'})
    if not r.get('ok'): raise RuntimeError(f"Node {nid} MAT_COMPUTE_D_E failed: {r}")
    d_parts.append(np.array(r['d_i'],dtype=np.uint64))
    e_parts.append(np.array(r['e_i'],dtype=np.uint64))
D_u,E_u=reconstruct(d_parts),reconstruct(e_parts)
D,E=D_u.view(np.int64).astype(np.float64),E_u.view(np.int64).astype(np.float64)
DE=np.round(D.dot(E)).astype(np.int64); DE_u=DE.view(np.uint64); DE_sh=share_uint64(DE_u)

for nid in range(3):
    msg={'type':'APPLY_BACTH_DE','D':D.tolist(),'E':E.tolist(),
         'DE_share':np.array(DE_sh[nid],dtype=np.uint64),
         'c_key':'c','a_key':'a','b_key':'b','store_as':'Z'}
    r=send(nid,msg)
    if not r.get('ok'):
        dbg=send(nid,{'type':'DEBUG_KEYS'})
        raise RuntimeError(f"Node {nid} APPLY_BACTH_DE failed: {r}; keys: {dbg}")

time.sleep(0.05)
Z_sh=[None]*3
for nid in range(3):
    r=send(nid,{'type':'GET','key':'Z'})
    if not r.get('ok'): raise RuntimeError(f"Node {nid} GET Z failed: {r}")
    Z_sh[nid]=np.array(r['share'],dtype=np.uint64)
Z_u=reconstruct(Z_sh); Z=Z_u.view(np.int64).astype(np.int64)
print("Reconstructed Z (int view):"); print(Z)
print("Expected A·B:"); print(A.dot(B))
