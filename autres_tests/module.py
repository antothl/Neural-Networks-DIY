import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        if y.shape[0] != yhat.shape[0]:
            raise ValueError(f"y batch size {y.shape[0]} does not match yhat batch size {yhat.shape[0]}")
        if y.shape[1] != yhat.shape[1]:
            raise ValueError(f"y dimension {y.shape[1]} does not match yhat dimension {yhat.shape[1]}")
        exp_shape = (y.shape[0], )
        loss = np.mean((y - yhat) ** 2, axis=1)
        assert loss.shape == exp_shape
        return loss

    def backward(self, y, yhat):
        batch_size = y.shape[0]
        dim_size = y.shape[1]
        if batch_size != batch_size:
            raise ValueError(f"y batch size {y.shape[0]} does not match yhat batch size {yhat.shape[0]}")
        if dim_size != dim_size:
            raise ValueError(f"y dimension {y.shape[1]} does not match yhat dimension {yhat.shape[1]}")
        exp_shape = (batch_size, dim_size)
        back = 2 * (yhat - y)
        assert back.shape == exp_shape
        return back

class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        if y.shape[0] != yhat.shape[0]:
            raise ValueError(f"y batch size {y.shape[0]} does not match yhat batch size {yhat.shape[0]}")
        if y.shape[1] != yhat.shape[1]:
            raise ValueError(f"y dimension {y.shape[1]} does not match yhat dimension {yhat.shape[1]}")
        exp_shape = (y.shape[0], )
        loss = -np.sum(y * np.log(yhat + 1e-10), axis=1)
        assert loss.shape == exp_shape
        return loss
    
    def backward(self, y, yhat):
        batch_size = y.shape[0]
        dim_size = y.shape[1]
        if batch_size != batch_size:
            raise ValueError(f"y batch size {y.shape[0]} does not match yhat batch size {yhat.shape[0]}")
        if dim_size != dim_size:
            raise ValueError(f"y dimension {y.shape[1]} does not match yhat dimension {yhat.shape[1]}")
        exp_shape = (batch_size, dim_size)
        back = - (y / (yhat + 1e-10))
        assert back.shape == exp_shape
        return back
    
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._parameters = np.random.randn(input_dim+1, output_dim)
        self._gradient = np.zeros((input_dim+1, output_dim))

    def forward(self, X):
        # print(f"X shape: {X.shape}, parameters shape: {self._parameters.shape}")
        x_with_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        if x_with_bias.shape[1] != self._parameters.shape[0]:
            raise ValueError(f"X dimension {x_with_bias.shape[1]} does not match parameter dimension {self._parameters.shape[0]}")
        out = np.dot(x_with_bias, self._parameters)
        assert out.shape[0] == X.shape[0]
        return out
    
    def backward_update_gradient(self, input, delta):
        # Ajout de la colonne de biais
        input_with_bias = np.concatenate([input, np.ones((input.shape[0], 1))], axis=1)
        self._gradient += np.dot(input_with_bias.T, delta)

    def backward_delta(self, input, delta):
    
        weight_no_bias = self._parameters[:-1, :]
        # On retire la dernière ligne de self._parameters pour ne pas inclure le biais
        # parce que autrement ça rajoute une dimension de delta 
        # print(f"input shape: {input.shape}, delta shape: {delta.shape}")
        # print(f"parameters.T shape: {weight_no_bias.T.shape}")

        return np.dot(delta, weight_no_bias.T )
    
    def zero_grad(self):
        self._gradient = np.zeros_like(self._gradient)
    
class Sigmoid(Module):
    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward_delta(self, input, delta):
        return delta * self.output * (1 - self.output)
    
    def update_parameters(self, gradient_step=0.001):
        pass  # Rien à faire, pas de paramètres

    def zero_grad(self):
        pass  # Pas de gradient à réinitialiser

    def backward_update_gradient(self, input, delta):
        pass  # Pas de gradient à accumuler
    
class Tanh(Module):
    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward_delta(self, input, delta):
        return delta * (1 - self.output ** 2)

    def update_parameters(self, gradient_step=0.001):
        pass

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass


class ReLU(Module):
    def forward(self, X):
        self.output = np.maximum(0, X)
        return self.output

    def backward_delta(self, input, delta):
        return delta * (self.output > 0).astype(delta.dtype)

    def update_parameters(self, gradient_step=0.001):
        pass

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

class Softmax(Module):
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output

    def backward_delta(self, input, delta):
        s = self.output.reshape(-1, 1)
        return delta * s * (1 - s)

    def update_parameters(self, gradient_step=0.001):
        pass

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    
class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.inputs = []  # pour stocker les entrées de chaque couche
        self._debug = False

    def forward(self, x):
        self.inputs = [x]
        for layer in self.layers:
            if self._debug:
                print(layer.__class__.__name__, x.shape, '(batch size, input_dim)')
            x = layer.forward(x)
            if self._debug:
                print('output shape:', x.shape)
            self.inputs.append(x)  # on garde les entrées de chaque couche pour la rétropropagation
        if self._debug:
            print('Forward pass done\n----------------')
        return x
    
    def backward_update_gradient(self, input, delta):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            input_i = self.inputs[i]
            if self._debug:
                print(layer.__class__.__name__, '(batch size, input_dim)')
                print('delta l+1 shape:', delta.shape)
            layer.backward_update_gradient(input_i, delta)
            delta = layer.backward_delta(input_i, delta)
            if self._debug:
                print('delta l shape:', delta.shape)
                print('input shape:', input_i.shape)


        return delta 

    def update_parameters(self, gradient_step=1e-3):
        for layer in self.layers:
            layer.update_parameters(gradient_step)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

