# triple_planner.py
import math

def plan(N, D, Dout, avg_deg, layers):
    # Matmul counts:
    # for layer: matmul h(NxD) * W(DxDout) -> N*D*Dout mults
    per_layer_matmul = N*D*Dout
    # for attention: dot per edge: avg_deg * N * D mults
    att_dot_mults = N * avg_deg * D
    # per-layer other matmuls: agg(NxD) * Wdest(DxDout): N*D*Dout
    total_mults = layers * (per_layer_matmul + att_dot_mults + per_layer_matmul)
    return {'per_layer_matmul': per_layer_matmul, 'att_dot_mults': att_dot_mults, 'total_mults': total_mults}

if __name__ == "__main__":
    N=1000; D=64; Dout=64; avg_deg=10; layers=2
    print(plan(N,D,Dout,avg_deg,layers))
