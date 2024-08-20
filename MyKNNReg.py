import numpy as np
import pandas as pd

class MyKNNReg():
    
    def __init__(self,
                 k: int = 3,
                 metric: str = 'euclidean',
                 weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight
        pass
    
    def __str__(self) -> str:
        return f'MyKNNReg class: k={self.k}'
    
    def fit(self,X: pd.DataFrame,y: pd.Series) -> None:
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = (X.shape[0], X.shape[1])
        
    def euclidean_metric(self, x_test: np.ndarray) -> float:
        return np.sqrt(np.sum((self.X - x_test) ** 2,axis=1))
    
    def euclidean_metric(self, x_test: np.ndarray) -> float:
        return np.sqrt(np.sum((self.X - x_test) ** 2,axis=1))
    
    def manhattan(self, x_test: np.ndarray) -> float:
        return np.sum(np.absolute(self.X - x_test), axis=1)
    
    def chebyshev(self, x_test: np.ndarray) -> float:
        return np.max(np.absolute(self.X - x_test),axis=1)
    
    def cos_metric(self, x_test: np.ndarray) -> float:
        return 1 - ((np.dot(self.X,x_test))/(np.sqrt(np.sum(self.X ** 2,axis=1)) * np.sqrt(np.sum(x_test ** 2))))

    def count_metric(self, x_test: np.ndarray):
        if self.metric == 'euclidean':
            return self.euclidean_metric(x_test)
        elif self.metric == 'chebyshev':
            return self.chebyshev(x_test)
        elif self.metric == 'manhattan':
            return self.manhattan(x_test)
        elif self.metric == 'cosine':
            return self.cos_metric(x_test)
    
    def make_prediction(self,x_test: np.ndarray) -> float:
        distances = self.count_metric(x_test)
        k_nearest = np.argsort(distances)[:self.k]
        neighbors = self.y[k_nearest]
        if self.weight == 'uniform':
            pred = np.mean(neighbors)
        elif self.weight == 'rank':
            ranks = np.arange(neighbors.size)
            Norm = np.sum(1 / (ranks + 1))
            weighted_vector = ((1 / (ranks + 1)) / Norm)
            pred = np.dot(weighted_vector , neighbors)
        elif self.weight == 'distance':
            Norm = np.sum(1 / (distances[k_nearest]))
            weighted_vector = (1 / (distances[k_nearest])) / Norm
            pred = np.dot(weighted_vector, neighbors)
            
        return pred
            
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_test = X_test.to_numpy()
        return np.array([self.make_prediction(x_test) for x_test in X_test])