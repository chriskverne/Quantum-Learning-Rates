import pennylane as qml

class VanillaSGD:
    def __init__(self, stepsize=0.1):
        self.stepsize = stepsize

    def step(self, objective_fn, params):
        gradients = qml.grad(objective_fn)(params)
        return params - self.stepsize * gradients