# mpc_node_worker_batched_fixed.py
import socket, threading, pickle, struct, argparse, numpy as np, traceback

HOST = 'localhost'
PORT_BASE = 9000

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
        self.store = {}  # key -> np.uint64 array (bit-patterns)

node_state = NodeState()

def uint64_to_int64(arr_u):
    return arr_u.view(np.int64)

def int64_to_uint64(arr_i):
    return arr_i.view(np.uint64)

FRAC_BITS = 16
SCALE = 1 << FRAC_BITS

def handle_req(conn, addr, node_id):
    try:
        req = recv_msg(conn)
        if req is None:
            conn.close(); return
        typ = req.get('type')

        if typ == 'PING':
            send_msg(conn, {'ok': True, 'node': node_id})
            return

        if typ == 'STORE':
            node_state.store[req['key']] = np.array(req['share'], dtype=np.uint64)
            send_msg(conn, {'ok': True})
            return

        if typ == 'GET':
            key = req['key']
            send_msg(conn, {'ok': True, 'share': node_state.store.get(key)})
            return

        if typ == 'DEBUG_KEYS':
            send_msg(conn, {'ok': True, 'keys': list(node_state.store.keys())})
            return

        if typ == 'MAT_COMPUTE_D_E':
            X = node_state.store.get(req['X_key']); Y = node_state.store.get(req['Y_key'])
            a = node_state.store.get(req['a_key']); b = node_state.store.get(req['b_key'])
            if X is None or Y is None or a is None or b is None:
                send_msg(conn, {'ok': False, 'err': 'missing X/Y/a/b keys'})
                return
            d_i = (X.astype(np.uint64) - a.astype(np.uint64)).astype(np.uint64)
            e_i = (Y.astype(np.uint64) - b.astype(np.uint64)).astype(np.uint64)
            send_msg(conn, {'ok': True, 'd_i': d_i, 'e_i': e_i})
            return

        if typ == 'APPLY_BACTH_DE_FIXED':
            # D, E are public (lists -> convert to float then to fixed?). Here we expect D,E already as integer (int64) lists representing fixed-point
            D_list = req['D']; E_list = req['E']
            DE_share = np.array(req['DE_share'], dtype=np.uint64)
            a = node_state.store.get(req['a_key']); b = node_state.store.get(req['b_key']); c = node_state.store.get(req['c_key'])
            if a is None or b is None or c is None:
                send_msg(conn, {'ok': False, 'err': 'missing a/b/c on node'})
                return

            # interpret a,b,c as int64 fixed-point values
            a_i = uint64_to_int64(np.array(a, dtype=np.uint64)).astype(np.int64)
            b_i = uint64_to_int64(np.array(b, dtype=np.uint64)).astype(np.int64)
            c_i = uint64_to_int64(np.array(c, dtype=np.uint64)).astype(np.int64)

            # public D,E are lists of numbers representing FIXED-POINT integers already (int), but coordinator sends them as python ints (we accept floats too)
            D = np.array(D_list, dtype=np.int64)
            E = np.array(E_list, dtype=np.int64)

            # sanity shape checks
            try:
                # compute D @ b_i  (D: MxK, b_i: KxN) -> MxN
                Db = D.dot(b_i)
                aE = a_i.dot(E)
            except Exception as ex:
                send_msg(conn, {'ok': False, 'err': f"shape error: D{D.shape} b{b_i.shape} a{a_i.shape} E{E.shape}; {ex}"})
                return

            # Db and aE are int64 products in scale^2 domain. We need to truncate by FRAC_BITS
            # perform rounding: (v + (1<<(FRAC_BITS-1))) >> FRAC_BITS  (arithmetic right-shift)
            add = (1 << (FRAC_BITS - 1))
            Db_trunc = np.right_shift(Db + add, FRAC_BITS).astype(np.int64)
            aE_trunc = np.right_shift(aE + add, FRAC_BITS).astype(np.int64)

            # assemble z_i = c_i + Db_trunc + aE_trunc + DE_share_i
            de_i = uint64_to_int64(np.array(DE_share, dtype=np.uint64)).astype(np.int64)
            z_i = c_i + Db_trunc + aE_trunc + de_i
            # clamp to int64 then store as uint64 bit-pattern
            z_i = np.clip(z_i, -(1<<63), (1<<63)-1).astype(np.int64)
            node_state.store[req['store_as']] = z_i.view(np.uint64)
            send_msg(conn, {'ok': True})
            return

        send_msg(conn, {'ok': False, 'err': 'unknown request type'})
    except Exception as ex:
        tb = traceback.format_exc()
        try:
            send_msg(conn, {'ok': False, 'err': f"{ex}; tb: {tb}"})
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
