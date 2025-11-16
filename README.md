# MPC-based LEX-GNN Implementation

This project implements a 3-party Multi-Party Computation (MPC) framework to securely run a Graph Neural Network (GNN) layer. The architecture is built from scratch in Python and uses a Coordinator/Worker model with Beaver triples for secure multiplication protocols.

The implementation is guided by the principles of the LEX-GNN paper (leveraging node probabilities to create differentiated messages) and is based on two local simulators:
* `lex_gnn_mpc_sim.py` (now deleted): A model that computes messages securely but performs attention publicly.
* `lex_gnn_mpc_no_recon.py`: A more advanced model that computes attention privately.

---

## 🚀 Current Status: GNN Layer v1 (Public Attention)

The current implementation successfully executes a **full GNN layer** based on the "public attention" model (formerly `lex_gnn_mpc_sim.py`).

1.  **Core MPC Framework:** A stable 3-party networked framework using a coordinator (`mpc_gnn_coordinator_v1.py`) and three workers (`mpc_node_worker_batched_fixed_v2.py`).

2.  **Complete MPC Primitives:** All core mathematical operations have been implemented as secure RPCs:
    * `SECURE_MATMUL` (Beaver triple-based)
    * `SECURE_HADAMARD_MUL` (Element-wise Beaver multiplication)
    * `PUBLIC_SCALAR_MUL` (Multiplying a private share by a public number)
    * `LOCAL_ADD` / `LOCAL_SUB` (Local share arithmetic, no communication)

3.  **Assembled GNN Layer (`mpc_gnn_coordinator_v1.py`):**
    * **Secure Message Computation:** Securely computes node messages `[m]` from `[h]`, `[W0]`, `[W1]`, and `[p]` using the `rpc_matmul`, `rpc_sub`, and `rpc_hadamard_mul` primitives.
    * **Public Attention:** Reconstructs the messages `m` on the coordinator, computes attention scores in the clear, and re-shares the aggregated result `[agg]` back to the nodes.
    * **Secure Final Projection:** Securely computes the final `[h_new]` from `[agg]` and `[h]` using `rpc_matmul` and `rpc_add`.

---

## Setting up the Environment

Before running any scripts, you need to set up the Python environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SampathLgd/mpc_based_lex_gnn.git](https://github.com/SampathLgd/mpc_based_lex_gnn.git)
    cd mpc_based_lex_gnn
    ```

2.  **Create a Python virtual environment:**
    ```bash
    # We use 'venv_lex' as seen in your command history
    python3 -m venv venv_lex
    ```

3.  **Activate the virtual environment:**
    ```bash
    source venv_lex/bin/activate
    ```
    *(Your terminal prompt should now show `(venv_lex)`)*

4.  **Install required packages:**
    ```bash
    # The project's core dependencies are numpy and pandas
    pip install numpy pandas
    ```

---

## ⚙️ How to Run

There are two main ways to run the code. Ensure your virtual environment is activated first (`source venv_lex/bin/activate`).

### 1. Run the Full GNN Layer (v1)

This script executes the complete GNN layer computation, including the public attention step, and compares the final result against a cleartext computation.

1.  Make the script executable:
    ```bash
    chmod +x run_gnn_v1.sh
    ```
2.  Run the coordinator and nodes:
    ```bash
    ./run_gnn_v1.sh
    ```




### 2. Run Individual MPC Primitive Tests

This script individually tests each of the core MPC primitives (`MATMUL`, `ADD`, `SUB`, `HADAMARD_MUL`, `PUBLIC_SCALAR_MUL`) to verify they are arithmetically correct.

1.  Make the script executable:
    ```bash
    chmod +x run_all_fixed.sh
    ```
2.  Run the tests:
    ```bash
    ./run_all_fixed.sh
    ```




---

## 📋 Next Steps

The framework is fully functional. The next major step is to implement the "private attention" model from `lex_gnn_mpc_no_recon.py`.

This involves:
1.  **Creating a `SECURE_DOT_PRODUCT` primitive.** This will likely be implemented by:
    * Adding new RPCs to the worker to extract a specific row from a shared matrix (e.g., `[m_u] = GET_ROW('m', u)`).
    * Orchestrating `rpc_matmul` on the two extracted row/column vectors `[m_u]` and `[m_v]`.
2.  **Opening the scalar result:** The `(1,1)` result of the dot product (`[score]`) will be reconstructed to a public value on the coordinator.
3.  **Using `rpc_scalar_mul`:** The public attention weights (now computed from the opened scores) will
