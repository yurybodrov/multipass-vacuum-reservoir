"""
ridge_readout.py
W_out Ridge-regression solver with metrics.
"""
import numpy as np

class RidgeReadout:
    def __init__(self, lambd: float = 1.0):
        self.lambd = lambd
        self.W_out = None
        self.X_mean = None
        self.Y_mean = None
        self.X_std = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        self.X_mean = np.mean(X_train, axis=1, keepdims=True)
        self.X_std = np.std(X_train, axis=1, keepdims=True) + 1e-8
        self.Y_mean = np.mean(Y_train, axis=1, keepdims=True)

        X_norm = (X_train - self.X_mean) / self.X_std
        Y_cent = Y_train - self.Y_mean

        N_feat = X_train.shape[0]
        cov = (X_norm @ X_norm.T) / X_train.shape[1]
        reg_matrix = cov + self.lambd * np.eye(N_feat)
        self.W_out = (Y_cent @ X_norm.T / X_train.shape[1]) @ np.linalg.inv(reg_matrix)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_norm = (X_test - self.X_mean) / self.X_std
        return self.Y_mean + self.W_out @ X_norm

def calc_wave_fidelity(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(np.vdot(y_ref, y_pred))**2
    den = np.vdot(y_ref, y_ref).real * np.vdot(y_pred, y_pred).real + 1e-15
    return float(num / den)

def calc_r2_score(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum(np.abs(y_ref - y_pred)**2)
    ss_tot = np.sum(np.abs(y_ref - np.mean(y_ref))**2) + 1e-15
    return float(max(0.0, 1.0 - ss_res / ss_tot))