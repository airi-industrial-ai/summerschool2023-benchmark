import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from fddbenchmark.dataloader import FDDDataloader
from .base import FaultDetectionModel


class FaultDetectionPCA(FaultDetectionModel):
    def __init__(self) -> None:
        super().__init__()

        self.scaler = StandardScaler()
        self.forest = IsolationForest()

    def fit(self, train_dataloader: FDDDataloader) -> None:
        x = np.concatenate([x for x, _, _ in train_dataloader])  # (dataset_size, window_size, input_dim)
        x = x.reshape(len(x), -1)  # (dataset_size, window_size * input_dim)

        x = self.scaler.fit_transform(x)
        self.forest.fit(x)

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
        return -self.forest.score_samples(x)
