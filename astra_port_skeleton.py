# astra_port_skeleton.py  (pseudo)
# This shows how to map each sim primitive -> ASTRA primitive

# 1) Input sharing: client encodes feature vectors into fixed-point integers,
# then uses ASTRA API to distribute additive shares to the 3 servers.
# Example:
client.send_shares_to_servers(shares)  # ASTRA equivalent

# 2) Beaver matmul in ASTRA:
# Server side code calls:
C_sh = astra.beaver_matmul(A_sh, B_sh)  # offline precomputed triples consumed

# 3) Secure scalar multiply:
C_sh = astra.scalar_mul(S_sh, M_sh)  # or perform matmul with column vector

# 4) Truncation:
C_sh_trunc = astra.trunc(C_sh, k=FRAC_BITS)

# 5) Opening scalars:
opened = astra.open_scalar(scalar_sh)

# 6) Offline triple precomputation:
astra.generate_triples(count = desired_count)

# 7) Worker loop outline:
while True:
    op = receive_rpc()
    if op.type == 'MATMUL':
        result = astra.beaver_matmul(op.A_sh, op.B_sh)
        send_response(result)
    elif op.type == 'OPEN':
        val = astra.open_scalar(op.sh)
        send_response(val)
