import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        if self._gradient is not None:
            self._gradient=np.zeros_like(self._parameters)

    def forward(self, X):
        raise NotImplementedError

    def update_parameters(self, gradient_step=1e-3):
        if self._parameters is not None and self._gradient is not None:
            self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._parameters = np.random.randn(input_dim + 1, output_dim) * np.sqrt(2 / (input_dim + output_dim))
        self._gradient = np.zeros((input_dim + 1, output_dim))

    def forward(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  
        return X @ self._parameters  

    def backward_update_gradient(self, input, delta):
        input = np.hstack([input, np.ones((input.shape[0], 1))])  
                
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)

        # Vérification après reshape
        print(f"delta reshaped shape: {delta.shape}")

        self._gradient += (input.T @ delta)


    def backward_delta(self, input, delta):
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)

        return delta @ self._parameters[:-1].T  

# Fonction de perte
class MSELoss(Loss):
    def forward(self, y, yhat):
        
        return (y - yhat) ** 2  

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, f"Dimension y : {y.shape}, yhat : {yhat.shape}"
        return (-2 * (y - yhat) / y.shape[0]).reshape(-1, 1)  


class TanH(Module):
    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward_delta(self, X, delta):
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)
        return delta * (1 - np.tanh(X) ** 2)  
    
    def backward_update_gradient(self, X, delta):
        pass 
    
    def update_parameters(self, lr):
        pass  

class Sigmoide(Module):
    def forward(self, X):
        self.X = X
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, X, delta):
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)
        sig = 1 / (1 + np.exp(-X))
        return delta * sig * (1 - sig) 
    
    def backward_update_gradient(self, X, delta):
        pass  
    
    def update_parameters(self, lr):
        pass  

class SimpleNN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.tanh = TanH()
        self.linear2 = Linear(hidden_dim, output_dim)
        self.sigmoid = Sigmoide()

    def forward(self, X):
        self.z1 = self.linear1.forward(X)     
        print(self.z1.shape)   
        self.a1 = self.tanh.forward(self.z1)  
        print(self.a1.shape)  
        self.z2 = self.linear2.forward(self.a1)
        print(self.z2.shape)  
        y_hat = self.sigmoid.forward(self.z2)
        print(y_hat.shape)  
    
        return y_hat

    def backward(self, X, y, loss):
        y_hat = self.forward(X)
        delta = loss.backward(y, y_hat)

        delta2 = self.sigmoid.backward_delta(self.z2, delta)  
        delta1 = self.linear2.backward_delta(self.a1, delta2)
        delta0 = self.tanh.backward_delta(self.z1, delta1) 

        self.linear2.backward_update_gradient(self.a1, delta2)
        self.linear1.backward_update_gradient(X, delta0)

    def update_parameters(self, lr):
        self.linear1.update_parameters(lr)
        self.linear2.update_parameters(lr)

    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()


class Sequentiel(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)
    
    def forward(self, X):
        self.inputs = []  # Stocker les entrées avant chaque module
        for module in self.modules:
            self.inputs.append(X)  # Stocke avant transformation
            X = module.forward(X)
        return X

    def backward(self, X, y, loss):
        y_hat = self.forward(X)
        delta = loss.backward(y, y_hat)

        # 1. Calculer d'abord tous les deltas
        deltas = [None] * len(self.modules)  
        for i in range(len(self.modules) - 1, -1, -1):
            deltas[i] = delta  
            delta = self.modules[i].backward_delta(self.inputs[i], delta)

        # 2. Mettre à jour les gradients pour les couches linéaires
        for i, module in enumerate(self.modules):
            if isinstance(module, Linear):  
                module.backward_update_gradient(self.inputs[i], deltas[i])

    def update_parameters(self, lr):
        for module in self.modules:
            module.update_parameters(lr)
    
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


