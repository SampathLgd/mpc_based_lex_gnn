# mpc_node_worker_batched_fixed_v2.py
# Worker: elementwise safe arithmetic for fixed-point batched Beaver apply
import socket, threading, pickle, struct, argparse, numpy as np, traceback

HOST = 'localhost'
PORT_BASE = 9000


# Robust small-share generator used by coordinator and workers for simulation
def share_uint64(x_uint):
    x = np.asarray(x_uint, dtype=np.uint64)
    rand_share_bound = 1 << 20
    s0 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s1 = np.random.randint(0, rand_share_bound, size=x.shape, dtype=np.uint64)
    s2_list=[]
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        xi = int(x[idx]); v0 = int(s0[idx]); v1 = int(s1[idx])
        s2_list.append((xi - v0 - v1) & ((1<<64)-1))
    s2 = np.array(s2_list, dtype=np.uint64).reshape(x.shape)
    return [s0, s1, s2]

def recv_msg(conn):
    raw = conn.recv(4)
    if not raw: return None
    (l,) = struct.unpack('!I', raw)
    data = b''
    while len(data) < l:
        chunk = conn.recv(l - len(data))
        if not chunk: break
        data += chunk
    return pickle.loads(data)

def send_msg(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack('!I', len(data)) + data)

class NodeState:
    def __init__(self):
        self.store = {}      # key -> np.uint64 array (bit-patterns)
        self.meta  = {}      # key -> dict{shape:..., dtype:...}

node_state = NodeState()

FRAC_BITS = 16
MASK64 = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
SCALE = 1 << FRAC_BITS

def np_as_uint64(x):
    arr = np.array(x, copy=True)
    if arr.dtype == np.uint64:
        out = arr
    elif arr.dtype == np.int64:
        out = arr.view(np.uint64)
    else:
        arr_i = arr.astype(np.int64)
        out = arr_i.view(np.uint64)
    return out

def recv_and_store(key, payload):
    arr_u = np_as_uint64(payload)
    node_state.store[key] = arr_u
    node_state.meta[key] = {'shape': tuple(arr_u.shape), 'dtype': str(arr_u.dtype)}
    return

def uint64_to_int64_arr(u):
    # returns int64 numpy array (signed)
    u = np.asarray(u, dtype=np.uint64)
    # elementwise convert to python ints then to int64 correctly
    it = np.nditer(u, flags=['multi_index'])
    out = np.empty(u.shape, dtype=np.int64)
    for _ in it:
        idx = it.multi_index
        v = int(u[idx])
        out[idx] = np.int64(v - (1 << 64) if v >= (1 << 63) else v)
    return out

def int64_to_uint64_arr(iarr):
    # clamp then view
    arr = np.clip(iarr, INT64_MIN, INT64_MAX).astype(np.int64)
    return arr.view(np.uint64)

def handle_req(conn, addr, node_id):
    try:
        req = recv_msg(conn)
        if req is None:
            conn.close(); return
        typ = req.get('type')

        if typ == 'PING':
            send_msg(conn, {'ok': True, 'node': node_id}); return

        if typ == 'STORE':
            key = req['key']; payload = req['share']
            recv_and_store(key, payload)
            send_msg(conn, {'ok': True}); return

        if typ == 'GET':
            key = req['key']
            send_msg(conn, {'ok': True, 'share': node_state.store.get(key), 'meta': node_state.meta.get(key)})
            return

        if typ == 'GET_SHAPE':
            key = req['key']
            meta = node_state.meta.get(key)
            send_msg(conn, {'ok': True, 'meta': meta})
            return

        if typ == 'DEBUG_KEYS':
            send_msg(conn, {'ok': True, 'keys_meta': node_state.meta}); return
        
        # --- NEW HANDLER ---
        if typ == 'LOAD_OFFLINE_DATA':
            filename = req['file']
            try:
                with open(filename, 'rb') as f:
                    offline_data = pickle.load(f)
                
                # Load all precomputed triples into the node's state
                for key, data in offline_data.items():
                    recv_and_store(key, data)
                
                print(f"Node {node_id} successfully loaded {len(offline_data)} keys from {filename}")
                send_msg(conn, {'ok': True, 'keys_loaded': len(offline_data)})
            except Exception as e:
                print(f"Node {node_id} FAILED to load {filename}: {e}")
                send_msg(conn, {'ok': False, 'err': str(e)})
            return
        # -------------------

        if typ == 'LOCAL_ADD':
            key_A = req['key_A']
            key_B = req['key_B']
            store_as = req['store_as']
            
            A_sh = node_state.store.get(key_A)
            B_sh = node_state.store.get(key_B)
            
            if A_sh is None or B_sh is None:
                send_msg(conn, {'ok': False, 'err': f'Missing key for LOCAL_ADD: A={key_A}, B={key_B}'})
                return
            
            C_sh = (A_sh.astype(np.uint64) + B_sh.astype(np.uint64)).astype(np.uint64)
            
            node_state.store[store_as] = C_sh
            node_state.meta[store_as] = {'shape': tuple(C_sh.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return
        
        if typ == 'LOCAL_SUB':
            key_A = req['key_A']
            key_B = req['key_B']
            store_as = req['store_as']
            
            A_sh = node_state.store.get(key_A)
            B_sh = node_state.store.get(key_B)
            
            if A_sh is None or B_sh is None:
                send_msg(conn, {'ok': False, 'err': f'Missing key for LOCAL_SUB: A={key_A}, B={key_B}'})
                return
            
            C_sh = (A_sh.astype(np.uint64) - B_sh.astype(np.uint64)).astype(np.uint64)
            
            node_state.store[store_as] = C_sh
            node_state.meta[store_as] = {'shape': tuple(C_sh.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return
        
        if typ == 'APPLY_HADAMARD_DE_FIXED':
            D_list = req['D']; E_list = req['E']
            DE_share = np.array(req['DE_share'], dtype=np.uint64)
            a_u = node_state.store.get(req['a_key'])
            b_u = node_state.store.get(req['b_key'])
            c_u = node_state.store.get(req['c_key'])
            
            if a_u is None or b_u is None or c_u is None:
                send_msg(conn, {'ok': False, 'err': 'missing a/b/c on node; keys:' + str(list(node_state.store.keys()))})
                return

            a_i = uint64_to_int64_arr(a_u)
            b_i = uint64_to_int64_arr(b_u)
            c_i = uint64_to_int64_arr(c_u)
            D = np.array(D_list, dtype=np.int64)
            E = np.array(E_list, dtype=np.int64)
            de_i = uint64_to_int64_arr(DE_share)

            if D.shape != a_i.shape or E.shape != b_i.shape or a_i.shape != c_i.shape or c_i.shape != de_i.shape:
                send_msg(conn, {'ok': False, 'err': f'shape mismatch: D{D.shape}, E{E.shape}, a{a_i.shape}, b{b_i.shape}, c{c_i.shape}, de{de_i.shape}'})
                return

            Db_i = (D.astype(np.int64) * b_i.astype(np.int64)).astype(np.int64)
            aE_i = (a_i.astype(np.int64) * E.astype(np.int64)).astype(np.int64)

            t_db = np.right_shift(Db_i, FRAC_BITS).astype(np.int64)
            t_aE = np.right_shift(aE_i, FRAC_BITS).astype(np.int64)
            
            z_i = c_i.astype(np.int64) + t_db + t_aE + de_i.astype(np.int64)
            z_i = np.clip(z_i, INT64_MIN, INT64_MAX)

            node_state.store[req['store_as']] = z_i.view(np.uint64)
            node_state.meta[req['store_as']] = {'shape': tuple(z_i.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return
        
        if typ == 'PUBLIC_SCALAR_MUL':
            key_in = req['key_in']
            scalar_fixed = np.int64(req['scalar_fixed'])
            store_as = req['store_as']
            
            X_sh_u = node_state.store.get(key_in)
            
            if X_sh_u is None:
                send_msg(conn, {'ok': False, 'err': f'Missing key for PUBLIC_SCALAR_MUL: {key_in}'})
                return
            
            X_sh_i = uint64_to_int64_arr(X_sh_u)
            prod_i = (X_sh_i.astype(np.int64) * scalar_fixed).astype(np.int64)
            trunc_i = np.right_shift(prod_i, FRAC_BITS).astype(np.int64)
            trunc_i = np.clip(trunc_i, INT64_MIN, INT64_MAX)
            
            node_state.store[store_as] = trunc_i.view(np.uint64)
            node_state.meta[store_as] = {'shape': tuple(trunc_i.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return
        
        if typ == 'EXTRACT_ROW':
            key_in = req['key_in']
            row_idx = int(req['row_idx'])
            store_as = req['store_as']
            
            X_sh_u = node_state.store.get(key_in)
            if X_sh_u is None:
                send_msg(conn, {'ok': False, 'err': f'Missing key for EXTRACT_ROW: {key_in}'})
                return
            
            row_share = X_sh_u[row_idx, :].reshape(1, -1)
            
            node_state.store[store_as] = row_share
            node_state.meta[store_as] = {'shape': tuple(row_share.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return

        if typ == 'EXTRACT_COL':
            key_in = req['key_in']
            row_idx = int(req['row_idx'])
            store_as = req['store_as']
            
            X_sh_u = node_state.store.get(key_in)
            if X_sh_u is None:
                send_msg(conn, {'ok': False, 'err': f'Missing key for EXTRACT_COL: {key_in}'})
                return
            
            col_share = X_sh_u[row_idx, :].reshape(-1, 1)
            
            node_state.store[store_as] = col_share
            node_state.meta[store_as] = {'shape': tuple(col_share.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return

        if typ == 'UPDATE_ROW':
            key_target = req['key_target']
            key_source = req['key_source']
            row_idx = int(req['row_idx'])
            
            target_sh = node_state.store.get(key_target)
            source_sh = node_state.store.get(key_source)
            
            if target_sh is None or source_sh is None:
                send_msg(conn, {'ok': False, 'err': f'Missing keys for UPDATE_ROW: T={key_target}, S={key_source}'})
                return
            
            if target_sh[row_idx, :].shape != source_sh.reshape(-1).shape:
                 send_msg(conn, {'ok': False, 'err': f'Shape mismatch: Target row {target_sh[row_idx, :].shape} vs Source {source_sh.reshape(-1).shape}'})
                 return

            target_sh[row_idx, :] = source_sh.reshape(-1)
            
            send_msg(conn, {'ok': True})
            return

        if typ == 'MAT_COMPUTE_D_E':
            X = node_state.store.get(req['X_key']); Y = node_state.store.get(req['Y_key'])
            a = node_state.store.get(req['a_key']); b = node_state.store.get(req['b_key'])
            if X is None or Y is None or a is None or b is None:
                send_msg(conn, {'ok': False, 'err': f'missing keys; present: {list(node_state.store.keys())}' })
                return
            d_i = (X.astype(np.uint64) - a.astype(np.uint64)).astype(np.uint64)
            e_i = (Y.astype(np.uint64) - b.astype(np.uint64)).astype(np.uint64)
            send_msg(conn, {'ok': True, 'd_i': d_i, 'e_i': e_i})
            return

        if typ == 'APPLY_BACTH_DE_FIXED':
            D_list = req['D']; E_list = req['E']
            DE_share = np.array(req['DE_share'], dtype=np.uint64)
            a_u = node_state.store.get(req['a_key']); b_u = node_state.store.get(req['b_key']); c_u = node_state.store.get(req['c_key'])
            if a_u is None or b_u is None or c_u is None:
                send_msg(conn, {'ok': False, 'err': 'missing a/b/c on node; keys:' + str(list(node_state.store.keys()))})
                return

            a_i = uint64_to_int64_arr(a_u)
            b_i = uint64_to_int64_arr(b_u)
            c_i = uint64_to_int64_arr(c_u)
            D = np.array(D_list, dtype=np.int64)
            E = np.array(E_list, dtype=np.int64)

            if D.ndim != 2 or E.ndim != 2:
                send_msg(conn, {'ok': False, 'err': f'public D/E must be 2D; got D{D.shape} E{E.shape}'})
                return
            a_shape = a_i.shape; b_shape = b_i.shape
            if D.shape[1] != b_shape[0] or a_shape[1] != E.shape[0] or a_shape[0] != D.shape[0] or b_shape[1] != E.shape[1]:
                send_msg(conn, {'ok': False, 'err': f'shape mismatch: D{D.shape} b{b_shape} a{a_shape} E{E.shape}'})
                return

            M = D.shape[0]; K = D.shape[1]; N = E.shape[1]
            Db = [[0]*N for _ in range(M)]
            aE = [[0]*N for _ in range(M)]
            for i in range(M):
                for j in range(N):
                    s_db = 0
                    s_aE = 0
                    for k in range(K):
                        s_db += int(D[i,k]) * int(b_i[k, j])
                        s_aE += int(a_i[i, k]) * int(E[k, j])
                    Db[i][j] = s_db
                    aE[i][j] = s_aE

            z_mat = [[0]*N for _ in range(M)]
            de_i = uint64_to_int64_arr(DE_share)
            for i in range(M):
                for j in range(N):
                    t_db = Db[i][j] >> FRAC_BITS
                    t_aE = aE[i][j] >> FRAC_BITS
                    val = int(c_i[i,j]) + int(t_db) + int(t_aE) + int(de_i[i,j])
                    if val < INT64_MIN: val = INT64_MIN
                    if val > INT64_MAX: val = INT64_MAX
                    z_mat[i][j] = val
            z_np = np.array(z_mat, dtype=np.int64)
            node_state.store[req['store_as']] = z_np.view(np.uint64)
            node_state.meta[req['store_as']] = {'shape': tuple(z_np.shape), 'dtype': 'uint64'}
            send_msg(conn, {'ok': True})
            return

        send_msg(conn, {'ok': False, 'err': 'unknown request type'})
    except Exception as ex:
        tb = traceback.format_exc()
        try:
            send_msg(conn, {'ok': False, 'err': f'{ex}; tb: {tb}'})
        except:
            pass
    finally:
        conn.close()

def server_loop(port, node_id):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, port))
    s.listen(8)
    print(f"Node {node_id} listening on {port}")
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle_req, args=(conn, addr, node_id), daemon=True).start()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=int, required=True)
    args = p.parse_args()
    server_loop(PORT_BASE + args.id, args.id)