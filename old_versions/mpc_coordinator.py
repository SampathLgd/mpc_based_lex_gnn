# mpc_coordinator.py (improved)
import socket, pickle, time, struct

HOST = 'localhost'
PORT_BASE = 9000

def send(node_id, msg, retries=10, backoff=0.2):
    port = PORT_BASE + node_id
    last_exc = None
    for attempt in range(retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, port))
            # send length-prefixed
            data = pickle.dumps(msg)
            s.sendall(struct.pack('!I', len(data)) + data)
            # read response
            raw_len = s.recv(4)
            if not raw_len:
                s.close()
                raise RuntimeError("no response length")
            (resp_len,) = struct.unpack('!I', raw_len)
            resp = b''
            while len(resp) < resp_len:
                chunk = s.recv(resp_len - len(resp))
                if not chunk:
                    break
                resp += chunk
            s.close()
            return pickle.loads(resp)
        except Exception as e:
            last_exc = e
            time.sleep(backoff)
            backoff *= 1.5
    raise last_exc

if __name__ == "__main__":
    try:
        resp = send(0, {'type': 'PING'})
        print("node0:", resp)
        resp = send(1, {'type': 'PING'})
        print("node1:", resp)
        resp = send(2, {'type': 'PING'})
        print("node2:", resp)
    except Exception as e:
        print("Coordinator error:", e)
