# mpc_node.py (improved)
import socket
import threading
import pickle
import argparse
import struct

PORT_BASE = 9000
HOST = 'localhost'

def recv_msg(conn):
    # read 4-byte length prefix then payload
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

def handle_conn(conn, addr, node_id):
    try:
        msg = recv_msg(conn)
        if msg is None:
            conn.close()
            return
        # simple handler
        if msg.get('type') == 'PING':
            send_msg(conn, {'ok': True, 'node': node_id})
        else:
            # echo back
            send_msg(conn, {'ok': True, 'echo': msg, 'node': node_id})
    except Exception as e:
        print(f"[node {node_id}] handler error: {e}")
    finally:
        conn.close()

def server_loop(port, node_id):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, port))
    s.listen(8)
    print(f"Node listening on {port}")
    try:
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_conn, args=(conn, addr, node_id), daemon=True).start()
    finally:
        s.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, required=True)
    args = parser.parse_args()
    port = PORT_BASE + args.id
    server_loop(port, args.id)
