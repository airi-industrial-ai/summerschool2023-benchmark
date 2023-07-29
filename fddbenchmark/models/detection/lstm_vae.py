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
            num_lstm_layers: int = 1,
            bidirectional: bool = True,
            beta: float = 0.1,
            device: str = 'cuda',
    ) -> None:
        super().__init__()

        self.scaler = StandardScaler()

        factor = 2 if bidirectional else 1
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_lstm_layers,
                               batch_first=True, bidirectional=bidirectional)
        self.fc_mean = nn.Linear(factor * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(factor * hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_lstm_layers,
                               batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(factor * hidden_dim, input_dim)

        self.beta = beta

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
            log_dir = log_dir / f'version_{version}'
            logger = SummaryWriter(log_dir)
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
                encoder_outputs = self.encoder(x)[0]  # (batch_size, window_size, hidden_dim)
                mean = self.fc_mean(encoder_outputs)  # (batch_size, window_size, latent_dim)
                std = torch.exp(self.fc_logvar(encoder_outputs) / 2)  # (batch_size, window_size, latent_dim)
                q_z = torch.distributions.Normal(mean, std)
                p_z = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
                decoder_outputs = self.head(self.decoder(q_z.rsample())[0])

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

        if log_dir is not None:
            torch.save(self.state_dict(), log_dir / 'last.pt')

    @torch.no_grad()
    def predict(
            self,
            x: np.ndarray,
            scoring: Literal['reconstruction_error', 'importance_sampling'] = 'reconstruction_error',
            num_mc_samples: int = 1000,
            std_x: float = 1.0
    ) -> np.ndarray:
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

        encoder_outputs = self.encoder(x)[0]  # (batch_size, window_size, hidden_dim)
        mean = self.fc_mean(encoder_outputs)  # (batch_size, window_size, latent_dim)
        std = torch.exp(self.fc_logvar(encoder_outputs) / 2)  # (batch_size, window_size, latent_dim)

        match scoring:
            case 'reconstruction_error':
                decoder_outputs = self.head(self.decoder(mean)[0])  # (batch_size, window_size, input_dim)
                recon_error = F.mse_loss(decoder_outputs, x, reduction='none').mean(dim=(1, 2))
                recon_error = recon_error.data.cpu().numpy()
                return recon_error
            case 'importance_sampling':
                q_z = torch.distributions.Normal(mean, std)
                p_z = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))

                z = q_z.sample(torch.Size([num_mc_samples]))  # (num_samples, batch_size, window_size, latent_dim)

                decoder_outputs = self.head(self.decoder(z.flatten(0, 1))[0]).reshape(num_mc_samples, *x.shape)
                p_x = torch.distributions.Normal(decoder_outputs, std_x)

                nll = math.log(num_mc_samples) - torch.logsumexp(
                    p_x.log_prob(x).mean(dim=(2, 3))
                    + p_z.log_prob(z).mean(dim=(2, 3))
                    - q_z.log_prob(z).mean(dim=(2, 3)),
                    dim=0
                )
                nll = nll.data.cpu().numpy()
                return nll
            case _:
                raise NotImplementedError(f"Scoring rule {scoring} is not supported.")
