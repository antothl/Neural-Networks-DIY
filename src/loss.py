import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError
    
class MSELoss(Loss):
    def forward(self, y, yhat):
        return (y - yhat) ** 2  

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, f"Dimension y : {y.shape}, yhat : {yhat.shape}"
        return (-2 * (y - yhat) / y.shape[0]).reshape(-1, 1)  
    
class CrossEntropyLoss:
    def forward(self, y_true, y_pred):
        epsilon = 1e-12 
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def backward(self, y_true, y_pred):
        return y_pred - y_true 
    
    def backward_update_gradient(self, X, delta):
        pass  
    
    def update_parameters(self, lr):
        pass 