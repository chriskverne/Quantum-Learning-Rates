import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# IMPORT YOUR CUSTOM OPTIMIZERS HERE
from decsps_opt import DecSPS
from sgd_opt import VanillaSGD


# 1. Environment Setup
num_qubits = 4
num_layers = 3
safe_l_star = float(1 - 2 * num_qubits)
dev = qml.device("default.qubit", wires=num_qubits)

# 2. Ansatz Configuration
def ansatz(p):
    for layer in range(num_layers):
        for qubit in range(num_qubits):
            qml.RX(p[layer, qubit, 0], wires=qubit)
            qml.RY(p[layer, qubit, 1], wires=qubit)
        for qubit in range(num_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

# 3. Hamiltonian Objective Configuration
obs = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(num_qubits - 1)] + [qml.PauliX(i) for i in range(num_qubits)]
coeffs = [-1.0] * len(obs)
Hamiltonian = qml.Hamiltonian(coeffs, obs)

@qml.qnode(dev)
def cost_function(params):
    ansatz(params)
    return qml.expval(Hamiltonian)

# 4. Parameters initialization
np.random.seed(42)
init_params = np.random.uniform(0, 2 * np.pi, size=(num_layers, num_qubits, 2), requires_grad=True)

# 5. Define your benchmark runs dictionary
benchmarks = {
    "DecSPS": DecSPS(l_star=safe_l_star, gamma_b=2.0), # Updated dynamically!
    "SGD": VanillaSGD(stepsize=0.1),
    "Adam": qml.AdamOptimizer(stepsize=0.1),
    "QNG": qml.QNGOptimizer(stepsize=0.01)
}

# 6. Execute loop
for name, optimizer in benchmarks.items():
    params = np.array(init_params.copy(), requires_grad=True)
    
    for step in range(101):
        params = optimizer.step(cost_function, params)
            


# =====================================================================
# EXACT CLASSICAL GROUND STATE CALCULATION (Add this before the loops!)
# =====================================================================
H_matrix = qml.matrix(Hamiltonian)
true_optimal_f = float(min(np.linalg.eigvalsh(H_matrix)))
print(f"Target Theoretical Ground Energy: {true_optimal_f:.6f}\n")

# =====================================================================
# 5. INITIALIZE STORAGE FOR PLOTTING DATA
# =====================================================================
# Maps each optimizer name to lists for recording step milestones
plot_data = {name: {"iterations": [], "energies": []} for name in benchmarks.keys()}

# =====================================================================
# 6. EXECUTE LOOP WITH STRIDED DATA COLLECTION
# =====================================================================
for name, optimizer in benchmarks.items():
    print(f"\n--- Running {name} Benchmark ---")
    params = np.array(init_params.copy(), requires_grad=True)
    
    for step in range(101):
        # Process the optimization update step
        params = optimizer.step(cost_function, params)
            
        # Capture data at every 10th milestone step (0, 10, 20... 100)
        if step % 10 == 0:
            current_energy = float(cost_function(params))
            
            # Record directly into memory arrays
            plot_data[name]["iterations"].append(step)
            plot_data[name]["energies"].append(current_energy)
            
            # Print to console for real-time tracking
            print(f"Iteration {step:3d} | Energy Cost: {current_energy:.6f}")

# =====================================================================
# 7. GENERATE THE MATPLOTLIB GRAPH
# =====================================================================
plt.figure(figsize=(10, 6))

# Loop through our collected data to plot each method's curve
for name, data in plot_data.items():
    plt.plot(
        data["iterations"], 
        data["energies"], 
        label=name, 
        marker='o',       # Places a distinct point marker every 10 steps
        linewidth=2
    )

# Add a dashed horizontal reference line for the theoretical true ground state
plt.axhline(
    y=true_optimal_f, 
    color='r', 
    linestyle='--', 
    label=f'True Ground Energy ({true_optimal_f:.4f})'
)

# Graph Layout & Dynamic Text Configuration
plt.title(f"Optimizer Comparison ({num_qubits} Qubits, {num_layers} Layers)", fontsize=14, fontweight='bold')
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Energy Expectation Value <H>", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Display the window containing the generated figure
plt.tight_layout()
plt.show()