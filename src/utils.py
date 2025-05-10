# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

class Optimizer:
    def __init__(self, model, loss, lr=0.01):
        self.model = model
        self.loss = loss
        self.lr = lr
        self.loss_log = []

    def step(self, X, y):
        self.model.zero_grad()
        # Forward pass
        yhat = self.model.forward(X)

        # Compute loss
        loss_value = self.loss.forward(y, yhat)
        self.loss_log.append(loss_value.mean())

        # Backward pass
        delta = self.loss.backward(y, yhat)
        self.model.backward_update_gradient(X, delta)
        self.model.update_parameters(self.lr)
        return loss_value.mean()
    
class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True, noise = None):
        if len(X) != len(y):
            raise ValueError(f"X size {len(X)} does not match y size {len(y)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise = noise
        self.index = 0

    def set_noise_function(self, noise):
        """
        Set the noise function to be applied to the data.

        :param noise: Function to apply noise
        """
        self.noise = noise
        return self

    def shuffle_set(self) -> "Dataset":
        """ 
        Shuffle the dataset 

        :return: self
        :rtype: Dataset
        """
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        self.index = 0
        return self

    def __len__(self):
        """
        Returns the number of batches in the dataset.

        :return: Number of batches
        :rtype: int
        """
        return len(self.X) // self.batch_size + (1 if len(self.X) % self.batch_size != 0 else 0)
    
    def __iter__(self):
        """
        Returns an iterator over the dataset.

        :return: Iterator over the dataset
        :rtype: Dataset
        """
        self.index = 0
        if self.shuffle:
            self.shuffle_set()
        return self
    
    def __next__(self):
        """
        Returns the next batch of data.

        :return: Tuple of (batch_X, batch_y)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if self.index >= len(self.X):
            raise StopIteration
        batch_X = self.X[self.index:self.index + self.batch_size].copy()
        batch_y = self.y[self.index:self.index + self.batch_size].copy()
        if self.noise is not None:
            batch_X = self.noise(batch_X)
        self.index += self.batch_size
        return batch_X, batch_y

class Trainer:
    def __init__(self
                 , model
                 , loss
                 , optimizer
                 , train_dataset
                 , val_dataset=None
                 , test_dataset=None
                 , epochs=10
                 , lr=0.01):
        """
        Initialize the Trainer class.

        :param model: Model instance
        :param loss: Loss function instance
        :param optimizer: Optimizer instance
        :param train_dataset: Dataset instance for training data
        :param val_dataset: Dataset instance for validation data
        :param test_dataset: Dataset instance for testing data
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer(model, loss, lr)
        self.epochs = epochs
        self.lr = lr
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train(self
              , metrics: list = None
              , verbose: bool = False
              , verbose_step: int = 1
              , valid_epoch: int = 1):
        """
        Train the model.

        :param epochs: Number of epochs to train
        :param metrics: List of metrics to evaluate
        """
        if metrics is None:
            metrics = []
        train_loss_log = []
        val_loss_log = []
        for epoch in range(self.epochs):
            loss = 0
            val_loss = 0
            for batch_X, batch_y in self.train_dataset:
                loss += self.optimizer.step(batch_X, batch_y)
            train_loss_log.append(loss/len(self.train_dataset))
            if verbose and (epoch + 1) % verbose_step == 0:
                feedback = f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss/len(self.train_dataset):.4f}"
                for metric in metrics:
                    metric_value = metric(batch_y, self.model.forward(batch_X))
                    feedback += f", {metric}: {metric_value:.4f}"
                print(feedback)
            if self.val_dataset and (epoch + 1) % valid_epoch == 0:
                for batch_X, batch_y in self.val_dataset:
                    val_loss += self.optimizer.step(batch_X, batch_y)
                val_loss_log.append(val_loss/len(self.val_dataset))
                if verbose:
                    feedback = f"Validation Loss: {val_loss/len(self.val_dataset):.4f}"
                    for metric in metrics:
                        metric_value = metric(batch_y, self.model.forward(batch_X))
                        feedback += f", {metric}: {metric_value:.4f}"
                    print(feedback)
            else:
                val_loss_log.append(None)
        return train_loss_log, val_loss_log
    
    def test(self, metrics: list = None):
        """
        Test the model.

        :param metrics: List of metrics to evaluate
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset provided.")
        if metrics is None:
            metrics = []
        feedback = f"Test "
        for metric in metrics:
            print(f"Testing {metric}")
            if self.test_dataset.noise is not None:
                batch_X = self.test_dataset.noise(self.test_dataset.X)
            else:
                batch_X = self.test_dataset.X
            metric_value = metric(self.test_dataset.y, self.model.forward(batch_X))
            feedback += f" {metric}: {metric_value:.4f}"
        print(feedback)



class Metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, y, yhat):
        pass

    def __str__(self):
        return self.name
    
class Accuracy(Metric): 
    def __init__(self):
        super().__init__("Accuracy")

    def __call__(self, y, yhat):
        if y.shape != yhat.shape:
            raise ValueError(f"y shape {y.shape} does not match yhat shape {yhat.shape}")
        yhat = np.where(yhat > 0.5, 1, 0)
        return accuracy_score(y, yhat)
    
class F1Score(Metric):
    def __init__(self):
        super().__init__("F1 Score")

    def __call__(self, y, yhat):
        if y.shape != yhat.shape:
            raise ValueError(f"y shape {y.shape} does not match yhat shape {yhat.shape}")
        yhat = np.where(yhat > 0.5, 1, 0)
        return f1_score(y, yhat)
    
class AUC(Metric):
    def __init__(self):
        super().__init__("AUC")

    def __call__(self, y, yhat):
        if y.shape != yhat.shape:
            raise ValueError(f"y shape {y.shape} does not match yhat shape {yhat.shape}")
        return roc_auc_score(y, yhat)
    
class LossMetric(Metric):
    def __init__(self, loss):
        super().__init__(loss.__class__.__name__)
        self.loss = loss

    def __call__(self, y, yhat):
        if y.shape != yhat.shape:
            raise ValueError(f"y shape {y.shape} does not match yhat shape {yhat.shape}")
        return self.loss.forward(y, yhat)
    
    def __str__(self):
        return f"{self.name} (Loss)"
    
def get_dataset_split(X, y, test_size=0.15, val_size=0.15, batch_size=32):
    """
    Split the dataset into train, validation, and test sets.

    :param X: Feature matrix
    :param y: Target vector
    :return: Tuple of (train_dataset, val_dataset, test_dataset)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted)
    
    train_dataset = Dataset(X_train, y_train, batch_size=batch_size)
    val_dataset = Dataset(X_val, y_val, batch_size=batch_size)
    test_dataset = Dataset(X_test, y_test, batch_size=batch_size)

    
    return train_dataset, val_dataset, test_dataset