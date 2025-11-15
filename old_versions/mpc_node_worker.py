# mpc_node_worker.py  (final corrected)
import socket, threading, pickle, struct, argparse, numpy as np

HOST = 'localhost'
PORT_BASE = 9000

def recv_msg(conn):
    raw_len = conn.recv(4)
    if not raw_len:
        return None
    (msg_len,) = struct.unpack('!I', raw_len)
    data = b''
    while len(data) < msg_len:
        chunk = conn.recv(msg_len - len(data))
        if not chunk:
            break
        data += chunk
    return pickle.loads(data)

def send_msg(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack('!I', len(data)) + data)

class NodeState:
    def __init__(self):
        self.store = {}  # map from key -> numpy uint64 array share

node_state = NodeState()

def handle_req(conn, addr, node_id):
    try:
        req = recv_msg(conn)
        if req is None:
            conn.close(); return
        typ = req.get('type')
        if typ == 'PING':
            send_msg(conn, {'ok': True, 'node': node_id})
            return

        if typ == 'STORE_SHARES':
            key = req['key']
            arr = req['share']
            node_state.store[key] = np.array(arr, dtype=np.uint64)
            send_msg(conn, {'ok': True})
            return

        if typ == 'GET_SHARE':
            key = req['key']
            arr = node_state.store.get(key)
            send_msg(conn, {'ok': True, 'share': arr})
            return

        if typ == 'COMPUTE_D_E':
            x = node_state.store[req['x_key']]
            y = node_state.store[req['y_key']]
            a = node_state.store[req['a_key']]
            b = node_state.store[req['b_key']]
            d_i = (x.astype(np.uint64) - a.astype(np.uint64)).astype(np.uint64)
            e_i = (y.astype(np.uint64) - b.astype(np.uint64)).astype(np.uint64)
            send_msg(conn, {'ok': True, 'd_i': d_i, 'e_i': e_i})
            return

        if typ == 'APPLY_DE_AND_DE_SHARE':
            # req: 'd' (public float scalar), 'e' (public float scalar), 'de_share' (this node's uint64 array),
            # keys: c_key, a_key, b_key, store_as
            d = float(req['d'])
            e = float(req['e'])
            de_share = np.array(req['de_share'], dtype=np.uint64)
            c = node_state.store[req['c_key']]
            a = node_state.store[req['a_key']]
            b = node_state.store[req['b_key']]

            # safer multiplication: interpret secret uint as signed int64 then to float and clamp
            def mul_public(secret_uint, public_float):
                secret_int = secret_uint.view(np.int64).astype(np.float64)
                CLAMP = 1e8
                secret_float = np.clip(secret_int, -CLAMP, CLAMP)
                prod = np.round(secret_float * public_float).astype(np.int64)
                prod = np.clip(prod, -(1<<63), (1<<63)-1).astype(np.int64).view(np.uint64)
                return prod

            term_db = mul_public(b, d)
            term_ea = mul_public(a, e)

            z_i = (c.astype(np.uint64) + term_db.astype(np.uint64) + term_ea.astype(np.uint64) + de_share.astype(np.uint64)).astype(np.uint64)
            node_state.store[req['store_as']] = z_i
            send_msg(conn, {'ok': True})
            return

        if typ == 'DEBUG_KEYS':
            send_msg(conn, {'ok': True, 'keys': list(node_state.store.keys())})
            return

        send_msg(conn, {'ok': False, 'err': 'unknown request type'})
    except Exception as ex:
        try:
            send_msg(conn, {'ok': False, 'err': str(ex)})
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=int, required=True)
    args = p.parse_args()
    server_loop(PORT_BASE + args.id, args.id)
