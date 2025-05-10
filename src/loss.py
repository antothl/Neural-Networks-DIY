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
        max_yhat = np.max(yhat, axis=1, keepdims=True)  
        logsumexp = np.log(np.sum(np.exp(yhat - max_yhat), axis=1, keepdims=True))
        log_softmax = yhat - max_yhat - logsumexp 
        return -np.sum(y * log_softmax) / y.shape[0]  

    def backward(self, y, yhat):
        max_yhat = np.max(yhat, axis=1, keepdims=True) 
        exp_yhat = np.exp(yhat - max_yhat)
        softmax = exp_yhat / np.sum(exp_yhat, axis=1, keepdims=True) 
        return softmax - y  

class BinaryCrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return (yhat - y) / (yhat * (1 - yhat)) / y.shape[0]
    
class BCEWeight(Loss):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.w = pos_weight

    def forward(self, y, yhat):
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(yhat) * self.w + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return (-y * self.w * (1 - yhat) - (1 - y) * yhat) / (yhat * (1 - yhat)) / y.shape[0]
