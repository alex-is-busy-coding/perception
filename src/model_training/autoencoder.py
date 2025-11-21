import torch
import torch.nn as nn
import pytorch_lightning as pl


class Conv1D_AE(nn.Module):
    """
    1D Convolutional Autoencoder.

    Includes BatchNorm for stabilization and Dropout for regularization.
    
    Assumes input shape is (batch_size, n_features, seq_len)
    """
    def __init__(self, n_features, latent_dim, dropout_prob=0.3):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(64, self.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(32, n_features, kernel_size=7, stride=2, padding=3, output_padding=1)
        )
    
    def forward(self, x):
        """
        Full reconstruction pass.
        Assumes x is already (B, n_features, S_padded) from pad_collate
        """
        original_seq_len = x.shape[2]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded[:, :, :original_seq_len]
        return decoded

    def get_latent_representation(self, x):
        """
        Helper function to get the flat latent representation.
        """
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        return encoded_flat


class LitConv1D_AE(pl.LightningModule):
    def __init__(self, n_features, latent_dim, learning_rate=1e-3, weight_decay=1e-5, dropout_prob=0.3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Conv1D_AE(
            n_features=self.hparams.n_features, 
            latent_dim=self.hparams.latent_dim,
            dropout_prob=self.hparams.dropout_prob
        )
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        """
        The forward pass for inference (reconstruction).
        """
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        sequences, mask = batch

        reconstructed = self.forward(sequences)

        per_element_loss = self.criterion(reconstructed, sequences)
        
        mask_expanded = mask.unsqueeze(1).expand_as(per_element_loss)
        
        masked_loss = per_element_loss * mask_expanded
        
        loss = masked_loss.sum() / (mask_expanded.sum() + 1e-8)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configure the optimizer.
        We add 'weight_decay' here for L2 regularization.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
