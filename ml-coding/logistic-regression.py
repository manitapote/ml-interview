import numpy as np
from typing import Tuple

def data(n_samples: int, n_features: int, seed: int = 42) -> Tuple[np.array, np.array]:
    """
    Generates synthetic data for logistic regression.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features for each sample.
        seed (int): Seed for random number generator.

    Returns:
        X (numpy.ndarray): Generated feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Generated binary labels of shape (n_samples,).1
    """
    np.random.seed(seed)
    
    X = np.random.random(size=(n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)
    
    return X, y

def scale(x: np.array) -> np.array:
    """
    Z score scaling
    
    Args:
        x (np.array): data of size (n_samples, n_features)
        
    Returns:
        x (np.array): scaled data (n_samples, n_features)
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    std = np.where(std==0, 1, std)
    
    return (x-mean)/std
    

def initialize_weight_bias(n_features: int) -> Tuple[np.array, float]:
    """
    Initialize weight and bias for first.
    
    Args:
        n_samples (int): number of samples
        n_features (int): number of feature
        
    Returns:
        w: (n_samples, n_features) weight
        b: (float) bias
    """
    w = np.random.random(size=(n_features, 1))
    b = float(np.random.randn())
    
    return w, b

def predict(x: np.array, w: np.array, b:float)->np.array:
    """
    Predicts the
    """
    #x=(n_samples, n_features)
    #w=(n_features, 1)
    #b=float
    
    #x.w = (n_samples, 1)
    #x.w + b = (n_samples, 1)
    
    return np.dot(x, w)+b
    
def sigmoid(y_pred: np.array) -> np.array:
    return 1/(1+ np.exp(-y_pred))

def logloss(y_pred: np.array, y: np.array, esp: float=1e-6) -> float:
    # mean of it ylog(y_pred)+(1-y)log(1-y_pred)
    #if y_pred is 0, log(y_pred) becomes infinity
    
    y_pred = np.clip(y_pred, esp, 1-esp)
    
    return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    
def gradient(x: np.array, y_pred: np.array, y: np.array) -> Tuple[float, float]:
    #dl/dw = (1/n)*x*(y_pred - y)
    #dl/db = 1/n * (y_pred - y) 
    
    #(n_samples)
    #x = (n_samples, n_features)
    #x.T.error = (n_features, n_samples). (n_samples)
    #dw=(n_features)
    error = y_pred - y
    dw = np.dot(x.T, error) *1/x.shape[0]
    db = float(np.mean(error))
    return dw, db

def update(w: np.array, learning_rate: float, dw: np.array, db: float, b: float):
    #w = (n_features, 1)
    #dw = (n_features, 1)
    return w - learning_rate*dw, b-learning_rate*db

if __name__ == "__main__":
    n_samples = 100
    n_features = 10
    seed = 42
    learning_rate = 0.1
    iteration = 1000

    # X = (n_samples, n_features)
    # y = (n_samples,)
    X, y = data(n_samples, n_features, seed)

    # X = (n_samples, n_features)
    X = scale(X)

    #w = (n_features, 1)
    #b = float
    w, b = initialize_weight_bias(n_features)
    
    result = []
    prev_loss = float('inf')
    est = 1e-6
    for i in range(iteration):
        #y_pred = 
        y_pred = predict(X, w, b)
        y_pred = y_pred.squeeze()
       
        y_pred = sigmoid(y_pred)
        loss = logloss(y_pred, y)
        
        dw, db = gradient(X, y_pred, y)
        dw = dw[:, np.newaxis]
        
        w, b = update(w, learning_rate, dw, db, b)
       
        result.append([i, loss])
        
        if prev_loss != float('inf') and abs(prev_loss - loss) < est:
            print('Converged', w.shape)
            break

        prev_loss = loss
        
        
        
        
        
    
    
    
