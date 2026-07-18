import pennylane as qml
import pennylane.numpy as np

# comapre to sgd with idk step size of 0.01 or adam with sgtep size 0.01 or qng
# test it on QAOA, VQE (different hamiltonians) and 2-15 qubit

# 1. Hyperparameters
num_qubits = 8
num_layers = 3

dev = qml.device("default.qubit", wires=num_qubits)

# 2. General 1D Transverse Field Ising Model (TFIM)
# H = - sum_{i=0}^{n-2} (Z_i Z_{i+1}) - sum_{i=0}^{n-1} (X_i)
coeffs = []
observables = []

# Nearest-neighbor Z interactions
for i in range(num_qubits - 1):
    coeffs.append(-1.0)
    observables.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
    
# Transverse X field on all qubits
for i in range(num_qubits):
    coeffs.append(-1.0)
    observables.append(qml.PauliX(i))

H = qml.Hamiltonian(coeffs, observables)

# 3. Generalized Hardware-Efficient Ansatz
def ansatz(params):
    # params shape: (num_layers, num_qubits, 2)
    for layer in range(num_layers):
        # Rotation block
        for q in range(num_qubits):
            qml.RX(params[layer, q, 0], wires=q)
            qml.RY(params[layer, q, 1], wires=q)
            
        # Entanglement block (linear chain)
        for q in range(num_qubits - 1):
            qml.CNOT(wires=[q, q+1])

# 4. QNodes
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params)
    return qml.expval(H)

@qml.qnode(dev)
def var_fn(params):
    ansatz(params)
    return qml.var(H) 

grad_fn = qml.grad(cost_fn)

# 5. Initialization
np.random.seed(42)
# Start in the same "flat" region by setting all initial angles to 2.0
params = np.ones((num_layers, num_qubits, 2), requires_grad=True) * 2.0

max_iterations = 60
epsilon = 1e-8 

print(f"--- Variance-Bounded Polyak Descent ---")
print(f"System: {num_qubits} Qubits | {num_layers} Layers | {params.size} Parameters\n")

# 6. Optimization Loop
for i in range(max_iterations):
    energy = cost_fn(params)
    variance = var_fn(params)
    
    gradient = grad_fn(params)
    
    # np.sum(gradient ** 2) gracefully handles the 3D tensor shape
    grad_norm_sq = np.sum(gradient ** 2) 
    
    step_size = np.sqrt(variance) / (grad_norm_sq + epsilon)
    
    # Update tensor
    params = params - step_size * gradient
    
    if (i + 1) % 5 == 0 or i == 0:
        print(f"Step {i+1:2d} | Energy: {energy:7.4f} | Var: {variance:7.4f} | Step Size: {step_size:7.4f}")

print("\nOptimization Complete.")
print(f"Final Estimated Ground State Energy: {energy:.6f}")

#estimate delta
#temple v.s. weinstein