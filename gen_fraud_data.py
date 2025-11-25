# gen_fraud_data.py
import numpy as np
import pickle

def generate_mini_fraud_dataset():
    print("--- Generating Mini Fraud Dataset ---")
    
    # 1. Configuration
    N = 20          # Total nodes (Keep it small for MPC testing)
    D = 16          # Feature dimension (Paper uses 32 for Yelp, we use 16 for speed) 
    Dout = 8        # Hidden/Output dimension
    fraud_ratio = 0.2 # 20% fraudsters (Paper has ~14.5% for Yelp) 
    
    np.random.seed(100) # Fixed seed for reproducibility

    # 2. Generate Features (h)
    # Fraudsters usually have different feature distributions than benign nodes
    n_fraud = int(N * fraud_ratio)
    n_benign = N - n_fraud
    
    # Benign nodes: Mean 0, Std 1
    h_benign = np.random.normal(0, 1, (n_benign, D))
    # Fraud nodes: Mean 2, Std 1.5 (Simulating "camouflaged" but distinct patterns) [cite: 7]
    h_fraud = np.random.normal(2, 1.5, (n_fraud, D))
    
    h = np.vstack([h_benign, h_fraud])
    
    # Labels (Y): 0 = Benign, 1 = Fraud
    Y = np.vstack([np.zeros((n_benign, 1)), np.ones((n_fraud, 1))])
    
    # 3. Generate Graph Topology (Adjacency)
    # Fraudsters tend to connect to other fraudsters (Homophily/Collusion)
    adjacency = {i: [] for i in range(N)}
    
    # Randomly connect nodes, but make fraudsters slightly more likely to connect to each other
    for i in range(N):
        # Each node connects to ~3 neighbors
        n_neighbors = np.random.randint(2, 5)
        
        if i >= n_benign: # If 'i' is a Fraudster
            # Prefer connecting to other fraudsters (indices n_benign to N)
            probs = np.ones(N)
            probs[n_benign:] *= 5 # 5x more likely to connect to fraud
            probs[i] = 0 # No self-loops here
            probs /= probs.sum()
            neighbors = np.random.choice(N, n_neighbors, p=probs, replace=False)
        else: # Benign
            probs = np.ones(N)
            probs[i] = 0
            probs /= probs.sum()
            neighbors = np.random.choice(N, n_neighbors, p=probs, replace=False)
            
        adjacency[i] = neighbors.tolist()

    # 4. Save to file
    data = {
        'N': N,
        'D': D,
        'Dout': Dout,
        'h': h,
        'Y': Y,
        'adjacency': adjacency,
        # We still need random weights since we haven't trained a model
        'W0': np.random.randn(D, Dout) * 0.1,
        'W1': np.random.randn(D, Dout) * 0.1,
        'Wself': np.random.randn(D, Dout) * 0.1,
        'Wdest': np.random.randn(Dout, Dout) * 0.1,
        'Psi0': np.random.randn(1, D) * 0.1,
        'Psi1': np.random.randn(1, D) * 0.1,
        'W_mlp1': np.random.randn(D, 8) * 0.1,
        'W_mlp2': np.random.randn(8, 1) * 0.1
    }
    
    with open('mini_fraud_data.pkl', 'wb') as f:
        pickle.dump(data, f)
        
    print(f"✅ Dataset saved to mini_fraud_data.pkl (N={N}, D={D})")

if __name__ == "__main__":
    generate_mini_fraud_dataset()