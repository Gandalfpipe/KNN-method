import numpy as np
import pandas as pd

class MyKNNClf():
    
    def __init__(self,
                 k: int =3,
                 metric: str = 'euclidean',
                 weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight
        pass
    
    
    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"
    
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
    
    def fit(self,X: pd.DataFrame,y: pd.Series) -> None:
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = (X.shape[0], X.shape[1])
    
    def make_prediction(self,x_test: np.ndarray) -> int:
        distances = self.count_metric(x_test)
        k_nearest = np.argsort(distances)[:self.k]
        neighbors = self.y[k_nearest]
        P = np.array(np.where(neighbors == 1))
        N = np.array(np.where(neighbors == 0))
        if self.weight == 'uniform':
            if P.size > N.size:
                pred = 1
            elif N.size > P.size:
                pred = 0
            else:
                pred = 1
        elif self.weight == 'rank':
            P = np.sum(1 / (P + 1))
            N = np.sum(1 / (N + 1))
            Norm = N + P
            if (P / Norm) > (N / Norm):
                pred = 1
            elif (P / Norm) < (N / Norm):
                pred = 0
            else:
                pred = 1
        elif self.weight == 'distance':
            distances = distances[k_nearest]
            Norm = np.sum(1 / distances)
            P = np.sum(1 / distances[P])
            N = np.sum(1 / distances[N])
            if (P / Norm) > (N / Norm):
                pred = 1
            if (P / Norm) < (N / Norm):
                pred = 0
            else:
                pred = 1

        return pred
        
    def probability(self,x_test: np.ndarray) -> float:
        distances = self.count_metric(x_test)
        k_nearest = np.argsort(distances)[:self.k]
        neighbors = self.y[k_nearest]
        P = np.array(np.where(neighbors == 1))
        N = np.array(np.where(neighbors == 0))
        if self.weight == 'uniform':
            pred = P.size / (N.size + P.size)
        elif self.weight == 'rank':
            P = np.sum(1 / (P + 1))
            N = np.sum(1 / (N + 1))
            pred = P / (N + P)
        elif self.weight == 'distance':
            distances = distances[k_nearest]
            Norm = np.sum(1 / distances)
            P = np.sum(1 / distances[P])
            N = np.sum(1 / distances[N])
            pred = P / Norm

        return pred
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_test = X_test.to_numpy()
        return np.array([self.make_prediction(x_test) for x_test in X_test])
    
    def predict_proba(self,X_test: pd.DataFrame):
        X_test = X_test.to_numpy()
        return pd.Series([self.probability(x_test) for x_test in X_test])
    
    