from typing import Optional, Literal
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fddbenchmark.dataloader import FDDDataloader
from .base import FaultDetectionModel


Scoring = Literal['reconstruction_error', 'importance_sampling']


class FaultDetectionLSTMVAE(nn.Module, FaultDetectionModel):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            beta: float = 0.1,
            scoring: Scoring = 'reconstruction_error',
            num_important_samples: int = 1000,
            device: str = 'cuda',
    ) -> None:
        super().__init__()

        self.scaler = StandardScaler()

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

        self.beta = beta
        self.scoring = scoring
        self.num_important_samples = num_important_samples

        self.to(device)
        self.device = device
    
    def _fit_scaler(self, train_dataloader: FDDDataloader) -> None:
        x = np.concatenate([x for x, _, _ in train_dataloader])
        x = x.reshape(len(x), -1)
        self.scaler.fit(x)
    
    def _scale(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x.reshape(len(x), -1)).reshape(x.shape)

    def fit(
            self,
            train_dataloader: FDDDataloader,
            num_epochs: int,
            warmup_epochs: float = 0.01,
            lr: float = 3e-4,
            log_dir: Optional[str] = None
    ) -> None:
        self._fit_scaler(train_dataloader)

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=num_epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=warmup_epochs
        )

        if log_dir is not None:
            log_dir = Path(log_dir)
            existing_versions = [int(path.name.split('_')[1]) for path in log_dir.glob('version_*')]
            version = max(existing_versions) + 1 if existing_versions else 0
            logger = SummaryWriter(log_dir / f'version_{version}')
        else:
            logger = None

        global_step = 0
        for epoch in range(num_epochs):
            if logger is not None:
                epoch_logs = {'recon_loss': [], 'kl': [], 'total_loss': []}

            for x, _, _ in tqdm(train_dataloader, desc=f'Epoch {epoch}, training loop'):
                x = self._scale(x)
                x = torch.tensor(x, dtype=torch.float32, device=self.device)  # (batch_size, window_size, input_dim)

                optimizer.zero_grad()

                # forward pass
                encoder_outputs, (_, _), = self.encoder(x)  # (batch_size, window_size, hidden_dim)
                mean = self.fc_mean(encoder_outputs)  # (batch_size, window_size, latent_dim)
                std = torch.exp(self.fc_logvar(encoder_outputs) / 2)  # (batch_size, window_size, latent_dim)
                q_z = torch.distributions.Normal(mean, std)
                p_z = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
                decoder_outputs, (_, _) = self.decoder(q_z.rsample())
                decoder_outputs = self.head(decoder_outputs)

                # loss
                recon_loss = F.mse_loss(decoder_outputs, x)
                kl = torch.distributions.kl_divergence(q_z, p_z).mean()
                loss = recon_loss + self.beta * kl

                if logger is not None:
                    logs = {
                        'recon_loss': recon_loss.item(),
                        'kl': kl.item(),
                        'total_loss': loss.item()
                    }
                    for k, v in logs.items():
                        logger.add_scalar(f'train/{k}_step', v, global_step)
                        epoch_logs[k].append(v)

                    logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                global_step += 1

            if logger is not None:
                logger.add_scalar('epoch', epoch, global_step)
                for k, v in epoch_logs.items():
                    logger.add_scalar(f'train/{k}_epoch', sum(v) / len(v), global_step)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts anomaly score for each time series window in a given batch.

        Args:
            x (np.ndarray):
                Batch of time series windows.
                Array of size ``(batch_size, window_size, input_dim)``.

        Returns:
            np.ndarray: Predicted anomaly scores. Array of size ``(batch_size,)``
        """
        self.eval()

        x = self._scale(x)
        x = torch.tensor(x, dtype=torch.float32, device=self.device)  # (batch_size, window_size, input_dim)

        encoder_outputs, (_, _), = self.encoder(x)  # (batch_size, window_size, hidden_dim)
        mean = self.fc_mean(encoder_outputs)  # (batch_size, window_size, latent_dim)
        std = torch.exp(self.fc_logvar(encoder_outputs) / 2)  # (batch_size, window_size, latent_dim)

        if self.scoring == 'reconstruction_error':
            decoder_outputs, (_, _) = self.decoder(mean)  # (batch_size, window_size, hidden_dim)
            decoder_outputs = self.head(decoder_outputs)  # (batch_size, window_size, input_dim)
            recon_error = F.mse_loss(decoder_outputs, x, reduction='none').mean(dim=(1, 2))
            recon_error = recon_error.data.cpu().numpy()
            return recon_error
        elif self.scoring == 'importance_sampling':
            batch_size, window_size, latent_dim = mean.shape
            num_samples = self.num_important_samples

            q_z = torch.distributions.Normal(mean, std)
            p_z = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
            z = q_z.sample(torch.Size([num_samples]))  # (num_samples, batch_size, window_size, latent_dim)

            decoder_outputs, (_, _) = self.decoder(z.reshape(-1, window_size, latent_dim))
            decoder_outputs = self.head(decoder_outputs)  # (num_samples * batch_size, window_size, input_dim)
            decoder_outputs = decoder_outputs.reshape(num_samples, batch_size, window_size, -1)
            recon_errors = F.mse_loss(decoder_outputs, x, reduction='none').mean(dim=(2, 3))  # (num_samples, batch_size)

            nll = -torch.logsumexp(
                -recon_errors
                + p_z.log_prob(z).mean(dim=(2, 3))
                - q_z.log_prob(z).mean(dim=(2, 3))
            )
            nll += math.log(num_samples)
        else:
            raise NotImplementedError(f"Scoring rule {self.scoring} is not supported.")
