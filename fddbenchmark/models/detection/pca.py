from typing import Literal
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from fddbenchmark.dataloader import FDDDataloader
from .base import FaultDetectionModel


class FaultDetectionPCA(FaultDetectionModel):
    def __init__(
            self,
            normal_variance_ratio: float = 0.9,
            scoring: Literal['spe', 't2'] = 'spe'
    ) -> None:
        super().__init__()

        self.scaler = StandardScaler()
        self.pca = PCA(normal_variance_ratio)
        
        self.scoring = scoring

    def fit(self, train_dataloader: FDDDataloader) -> None:
        x = np.concatenate([x for x, _, _ in train_dataloader])  # (dataset_size, window_size, input_dim)
        x = x.reshape(len(x), -1)  # (dataset_size, window_size * input_dim)

        x = self.scaler.fit_transform(x)
        self.pca.fit(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts anomaly score for each time series window in a given batch.

        Args:
            x (np.ndarray):
                Batch of time series windows.
                Array of size ``(batch_size, window_size, input_dim)``.

        Returns:
            np.ndarray: Predicted anomaly scores. Array of size ``(batch_size,)``
        """
        x = x.reshape(len(x), -1)  # (batch_size, window_size * input_dim)
        x = self.scaler.transform(x)  # (batch_size, window_size * input_dim)
        x_pca = self.pca.transform(x)  # (batch_size, num_components)
        if self.scoring == 't2':
            return np.sum(x_pca ** 2 / self.pca.explained_variance_, axis=1)  # (batch_size,)
        elif self.scoring == 'spe':
            return np.linalg.norm(self.pca.inverse_transform(x_pca) - x, axis=1)
        else:
            raise NotImplementedError(f"Scoring rule {self.scoring} is not supported.")
