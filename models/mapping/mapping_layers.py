import torch
import torch.nn as nn

class MappingLayer(nn.Module):
    """Base mapping layer that projects embeddings into shared space."""
    def __init__(self, input_dim: int, shared_dim: int):
        """
        Args:
            input_dim: Dimension of input embeddings (EA or EB)
            shared_dim: Dimension of shared space (ZA or ZB)
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input embeddings to shared space.
        
        Args:
            x: Input embeddings tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Mapped embeddings in shared space of shape (batch_size, seq_len, shared_dim)
        """
        return self.projection(x)

class AudioMapper(MappingLayer):
    """Mapping layer MA for audio embeddings."""
    def __init__(self, audio_dim: int, shared_dim: int):
        super().__init__(audio_dim, shared_dim)

class EEGMapper(MappingLayer):
    """Mapping layer MB for EEG embeddings."""
    def __init__(self, eeg_dim: int, shared_dim: int):
        super().__init__(eeg_dim, shared_dim)

class SharedSpaceMapper(nn.Module):
    """
    Combined module that handles mapping both audio and EEG embeddings 
    to shared space and computing alignment loss.
    """
    def __init__(
        self, 
        audio_dim: int,
        eeg_dim: int, 
        shared_dim: int
    ):
        """
        Args:
            audio_dim: Dimension of audio embeddings (EA)
            eeg_dim: Dimension of EEG embeddings (EB)
            shared_dim: Dimension of shared space
        """
        super().__init__()
        self.audio_mapper = AudioMapper(audio_dim, shared_dim)
        self.eeg_mapper = EEGMapper(eeg_dim, shared_dim)
        
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        eeg_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps both audio and EEG embeddings to shared space.
        
        Args:
            audio_embeddings: Audio embeddings EA of shape (batch_size, seq_len, audio_dim)
            eeg_embeddings: EEG embeddings EB of shape (batch_size, seq_len, eeg_dim)
            
        Returns:
            Tuple of:
                - ZA: Mapped audio embeddings in shared space
                - ZB: Mapped EEG embeddings in shared space
        """
        za = self.audio_mapper(audio_embeddings)
        zb = self.eeg_mapper(eeg_embeddings)
        return za, zb 