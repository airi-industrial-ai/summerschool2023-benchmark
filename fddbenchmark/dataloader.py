from tqdm import tqdm
import pandas as pd
import numpy as np


class FDDDataloader():
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        mask: pd.Series, 
        labels: pd.Series,
        window_size: int, 
        step_size: int, 
        minibatch_training=False, 
        batch_size=0, 
        shuffle=False,
        random_state=None,
    ):
        assert batch_size if minibatch_training else True
        self.df = dataframe
        self.labels = labels
        assert np.all(self.labels.index == self.df.index)
        self.window_size = window_size
        self.step_size = step_size
        assert self.step_size <= self.window_size
        sample_seq = []
        runs = self.labels[mask].index.get_level_values(0).unique()
        for run_id in tqdm(
            runs, 
            desc='Creating sequence of samples'):
            _idx = self.labels.index.get_locs([run_id])
            sample_seq.extend(
                np.arange(_idx.min(), _idx.max() - self.window_size + 1, self.step_size)
            )
        if random_state is not None:
            np.random.seed(random_state)
        self.sample_seq = np.random.permutation(sample_seq) if shuffle else np.array(sample_seq)
        n_samples = len(sample_seq)
        batch_seq = list(range(0, n_samples, batch_size)) if minibatch_training else [0]
        if batch_seq[-1] < n_samples: 
            batch_seq.append(n_samples)
        self.n_batches = len(batch_seq) - 1
        self.batch_seq = np.array(batch_seq)
    
    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.n_batches:
            # preparing batch of labels
            sample_ids = self.sample_seq[self.batch_seq[self.iter]:self.batch_seq[self.iter+1]]
            row_idx = np.tile(sample_ids[:, None], (1, self.window_size)) + np.arange(self.window_size)
            row_isna = np.isnan(self.labels.values[row_idx]).min(axis=1)
            labels_batch = np.zeros(row_isna.shape[0])
            labels_batch[row_isna] = np.nan
            if ~row_isna.any():
                # maximum label reduction: if at least a single value is fault
                # then the entire sample is fault
                labels_batch[~row_isna] = self.labels.values[row_idx][~row_isna].max(axis=1)
            # an index of a sample is an index of the last time stamp in the sample
            index_batch = self.labels.index[row_idx.max(axis=1)]
            labels_batch = pd.Series(labels_batch, name='labels', index=index_batch)
            # preparing batch of time series
            row_idx = np.tile(row_idx[..., None], (1, 1, self.df.shape[1]))
            col_idx = np.arange(self.df.shape[1])[None, None, :]
            col_idx = np.tile(col_idx, (row_idx.shape[0], self.window_size, 1))
            ts_batch = self.df.values[row_idx, col_idx]
            self.iter += 1
            return ts_batch, labels_batch.index, labels_batch
        else:
            raise StopIteration
