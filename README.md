# MPC-based LEX-GNN Implementation

This project creates a privacy-preserving implementation of the **LEX-GNN (Label-Exploring Graph Neural Network)** using a custom 3-party Multi-Party Computation (MPC) framework.

It translates the theoretical mathematics of the LEX-GNN paper into a working distributed system where data remains encrypted (secret-shared) throughout the entire inference and optimization process.

---

## 🚀 Key Features

### 1. Full Architectural Implementation
The system implements the complete pipeline described in the LEX-GNN paper (Equations 1 through 9):
* **Label Exploration (Eq 1-2):** Secure MLP for fraud probability prediction and loss calculation.
* **Message Construction (Eq 3-5):** Exact implementation of Label Embeddings ($\Psi$) and weight interpolation.
* **Private Attention (Eq 6):** Secure computation of attention scores without reconstructing message vectors.
* **Differentiated Reception (Eq 7):** Exact interpolation of destination weights ($W_{d0}, W_{d1}$).
* **Optimization (Eq 8-9):** Secure final classification and total loss calculation.

### 2. Offline/Online Phase Separation
To achieve realistic performance, the architecture is split:
* **Offline Phase (`precompute.py`):** Generates cryptographic material (Beaver Triples) in advance.
* **Online Phase (`mpc_gnn_coordinator_v2.py`):** Uses pre-computed material for fast, secure inference.

### 3. Secure Primitives
A suite of custom MPC Remote Procedure Calls (RPCs) handles the math:
* `SECURE_MATMUL` (Matrix Multiplication via Beaver Triples)
* `SECURE_HADAMARD_MUL` (Element-wise Multiplication)
* `PUBLIC_SCALAR_MUL` (Private Vector × Public Scalar)
* `UPDATE_ROW` / `EXTRACT_ROW` (Secure Matrix Manipulation)

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/SampathLgd/mpc_based_lex_gnn.git](https://github.com/SampathLgd/mpc_based_lex_gnn.git)
cd mpc_based_lex_gnn
```
2. Create Virtual Environment
Create and activate a clean Python environment to manage dependencies.

Bash
```
# Create environment named 'venv_lex'
python3 -m venv venv_lex

# Activate it
source venv_lex/bin/activate
```
3. Install Dependencies
The project requires numpy for matrix operations and pandas for data handling.

Bash
```
pip install numpy pandas
```
⚙️ How to Run
The project includes an automated script that handles the entire workflow: data generation, offline pre-computation, and the secure online run.

Step 1: Generate Synthetic Fraud Data
Create a structured dataset with "Fraud" and "Benign" clusters to verify the GNN's learning capability.

Bash
```
python gen_fraud_data.py
Output: mini_fraud_data.pkl (Contains graph topology, features, and weights).
```
Step 2: Run the Secure Pipeline
Execute the master script. This script performs three actions sequentially:

Runs precompute.py to generate Beaver triples matching the dataset dimensions.

Launches 3 background Worker nodes (mpc_node_worker_batched_fixed_v2.py).

Runs the Coordinator (mpc_gnn_coordinator_v2.py) to execute the GNN.

Bash
```
chmod +x run_gnn_v2.sh
./run_gnn_v2.sh
```
📊 Understanding the Output
A successful run will display the following verification stages:

Secure Primitive Checks:


```
✅ Secure MLP complete. [p] calculated.
✅ Secure Loss Calculation complete.
✅ Exact secure message computation complete.
✅ Private attention complete.
✅ Secure final projection complete.
Mathematical Accuracy: The system compares the Secure MPC result against a local Cleartext reference implementation.
```

```
✅ GNN Layer v2 (Eq 1-9) PASSED! (Difference: 0.022...)
A small difference (e.g., < 0.05) is expected due to fixed-point arithmetic truncation (FRAC_BITS=16).

Fraud Detection Capability: The logs analyze the final private embeddings. Distinct values for Fraud vs. Benign nodes prove the secure logic correctly processed the input features.
```

```
Mean Embedding (Fraud): [ 1.83, -0.85, ... ]
Mean Embedding (Benign): [ 0.20, 0.23, ... ]
Distinct output patterns verify the GNN processed the diverse input features correctly.
```
📂 Project Structure
mpc_node_worker_batched_fixed_v2.py: The server process. Handles share storage, fixed-point arithmetic, and socket communication.

mpc_gnn_coordinator_v2.py: The client/master. Defines the sequence of GNN operations and orchestrates the 3 workers.

precompute.py: Generates random Beaver Triples (a, b, c) and saves them to disk for the workers (Offline Phase).

gen_fraud_data.py: Generates a synthetic graph dataset with homophily and class imbalance.

run_gnn_v2.sh: The automated entry point script.
