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
        return (-2 * (y - yhat) / y.shape[0]) 
    
class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        yhat = np.clip(yhat, 1e-12, 1.0) 
        return 1-np.sum(y * yhat) / y.shape[0] 

    def backward(self, y, yhat):
        yhat = np.clip(yhat, 1e-12, 1.0) 
        return (yhat - y) / y.shape[0]  
    
class LogCrossEntropyLoss(Loss):
    def forward(self, y, yhat):  
        max_yhat = np.max(yhat, axis=1, keepdims=True)  # Stabilisation numérique
        logsumexp = np.log(np.sum(np.exp(yhat - max_yhat), axis=1, keepdims=True))
        log_softmax = yhat - max_yhat - logsumexp  # LogSoftmax appliqué ici
        return -np.sum(y * log_softmax) / y.shape[0]  # Cross-Entropy avec LogSoftmax

    def backward(self, y, yhat):
        max_yhat = np.max(yhat, axis=1, keepdims=True)  # Stabilisation numérique
        exp_yhat = np.exp(yhat - max_yhat)
        softmax = exp_yhat / np.sum(exp_yhat, axis=1, keepdims=True)  # Softmax
        return softmax - y  # Gradient de la cross-entropy
