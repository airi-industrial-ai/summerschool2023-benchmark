from abc import ABC, abstractmethod

import numpy as np


class FaultDetectionModel(ABC):

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Fits the model on the training set.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts anomaly score for each time series window in a given batch.

        Args:
            x (np.ndarray):
                Batch of time series windows.
                Array of size ``(batch_size, window_size, input_dim)``.

        Returns:
            np.ndarray: Predicted anomaly scores. Array of size ``(batch_size,)``
        """
        raise NotImplementedError
