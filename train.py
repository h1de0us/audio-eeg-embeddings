import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple

from models.mapping.mapping_layers import SharedSpaceMapper
from models.mapping.predictor import Predictor
from src.utils.loss import CombinedLoss
from data.dataset import NMEDDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.shared_mapper = SharedSpaceMapper(
            audio_dim=self.config["model"]["audio_dim"],
            eeg_dim=self.config["model"]["eeg_dim"],
            shared_dim=self.config["model"]["shared_dim"]
        ).to(self.device)
        
        self.predictor = Predictor(
            shared_dim=self.config["model"]["shared_dim"],
            audio_dim=self.config["model"]["audio_dim"]
        ).to(self.device)
        
        # Initialize loss and optimizers
        self.criterion = CombinedLoss(
            lambda1=self.config["training"]["lambda1"],
            lambda2=self.config["training"]["lambda2"],
            alignment_loss_type=self.config["training"]["alignment_loss_type"],
            reconstruction_loss_type=self.config["training"]["reconstruction_loss_type"]
        )
        
        self.optimizer = optim.Adam([
            {'params': self.shared_mapper.parameters()},
            {'params': self.predictor.parameters()}
        ], lr=self.config["training"]["learning_rate"])
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._setup_dataloaders()
        
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize train and validation data loaders."""
        train_dataset = NMEDDataset(
            split="train",
            data_dir=self.config["data"]["data_dir"]
        )
        val_dataset = NMEDDataset(
            split="val",
            data_dir=self.config["data"]["data_dir"]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"]
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"]
        )
        
        return train_loader, val_loader
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.shared_mapper.train()
        self.predictor.train()
        
        epoch_losses = {
            "total_loss": 0.0,
            "alignment_loss": 0.0,
            "reconstruction_loss": 0.0
        }
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            audio_embeddings = batch["audio_embeddings"].to(self.device)
            eeg_embeddings = batch["eeg_embeddings"].to(self.device)
            
            # Forward pass through mapping layers
            za, zb = self.shared_mapper(audio_embeddings, eeg_embeddings)
            
            # Predict audio embeddings from mapped EEG
            predicted_audio = self.predictor(zb)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                za=za,
                zb=zb,
                predicted_audio_embeddings=predicted_audio,
                target_audio_embeddings=audio_embeddings
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update running losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}"
            })
        
        # Compute average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.shared_mapper.eval()
        self.predictor.eval()
        
        val_losses = {
            "total_loss": 0.0,
            "alignment_loss": 0.0,
            "reconstruction_loss": 0.0
        }
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            audio_embeddings = batch["audio_embeddings"].to(self.device)
            eeg_embeddings = batch["eeg_embeddings"].to(self.device)
            
            za, zb = self.shared_mapper(audio_embeddings, eeg_embeddings)
            predicted_audio = self.predictor(zb)
            
            _, loss_dict = self.criterion(
                za=za,
                zb=zb,
                predicted_audio_embeddings=predicted_audio,
                target_audio_embeddings=audio_embeddings
            )
            
            for key in val_losses:
                val_losses[key] += loss_dict[key]
        
        # Compute average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["training"]["num_epochs"]):
            logger.info(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_losses = self.train_epoch()
            logger.info(
                f"Training Losses - Total: {train_losses['total_loss']:.4f}, "
                f"Alignment: {train_losses['alignment_loss']:.4f}, "
                f"Reconstruction: {train_losses['reconstruction_loss']:.4f}"
            )
            
            # Validate
            val_losses = self.validate()
            logger.info(
                f"Validation Losses - Total: {val_losses['total_loss']:.4f}, "
                f"Alignment: {val_losses['alignment_loss']:.4f}, "
                f"Reconstruction: {val_losses['reconstruction_loss']:.4f}"
            )
            
            # Early stopping
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                patience_counter = 0
                self._save_checkpoint(epoch, val_losses['total_loss'])
            else:
                patience_counter += 1
                
            if patience_counter >= self.config["training"]["patience"]:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'shared_mapper_state_dict': self.shared_mapper.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(
            checkpoint,
            checkpoint_dir / f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
