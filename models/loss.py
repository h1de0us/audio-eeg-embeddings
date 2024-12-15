import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLoss(nn.Module):
    """
    Implements Loss_align that minimizes the distance between 
    mapped audio (ZA) and EEG (ZB) embeddings in shared space.
    """
    def __init__(self, distance_type: str = "cosine"):
        """
        Args:
            distance_type: Type of distance metric to use.
                Options: "cosine", "mse", "l1"
        """
        super().__init__()
        self.distance_type = distance_type

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between mapped embeddings.

        Args:
            za: Mapped audio embeddings of shape (batch_size, seq_len, shared_dim)
            zb: Mapped EEG embeddings of shape (batch_size, seq_len, shared_dim)

        Returns:
            Scalar loss value
        """
        if self.distance_type == "cosine":
            # Normalize embeddings
            za_norm = F.normalize(za, p=2, dim=-1)
            zb_norm = F.normalize(zb, p=2, dim=-1)
            # Compute cosine similarity and convert to distance
            similarity = torch.sum(za_norm * zb_norm, dim=-1)
            loss = 1 - similarity.mean()
        elif self.distance_type == "mse":
            loss = F.mse_loss(za, zb)
        elif self.distance_type == "l1":
            loss = F.l1_loss(za, zb)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return loss

class ReconstructionLoss(nn.Module):
    """
    Implements Loss_rec that measures how well the predictor P 
    reconstructs audio embeddings EA from mapped EEG embeddings ZB.
    """
    def __init__(self, loss_type: str = "mse"):
        """
        Args:
            loss_type: Type of reconstruction loss.
                Options: "mse", "l1", "huber"
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predicted_audio_embeddings: torch.Tensor,
        target_audio_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between predicted and target audio embeddings.

        Args:
            predicted_audio_embeddings: Predicted audio embeddings (E_A_hat)
                Shape: (batch_size, seq_len, audio_dim)
            target_audio_embeddings: Target audio embeddings (E_A)
                Shape: (batch_size, seq_len, audio_dim)

        Returns:
            Scalar loss value
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(
                predicted_audio_embeddings,
                target_audio_embeddings
            )
        elif self.loss_type == "l1":
            loss = F.l1_loss(
                predicted_audio_embeddings,
                target_audio_embeddings
            )
        elif self.loss_type == "huber":
            loss = F.huber_loss(
                predicted_audio_embeddings,
                target_audio_embeddings
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss

class CombinedLoss(nn.Module):
    """
    Combines alignment and reconstruction losses with weighting factors.
    total_loss = λ1 * Loss_align + λ2 * Loss_rec
    """
    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        alignment_loss_type: str = "cosine",
        reconstruction_loss_type: str = "mse"
    ):
        """
        Args:
            lambda1: Weight for alignment loss (λ1)
            lambda2: Weight for reconstruction loss (λ2)
            alignment_loss_type: Type of alignment loss
            reconstruction_loss_type: Type of reconstruction loss
        """
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alignment_loss = AlignmentLoss(alignment_loss_type)
        self.reconstruction_loss = ReconstructionLoss(reconstruction_loss_type)

    def forward(
        self,
        za: torch.Tensor,
        zb: torch.Tensor,
        predicted_audio_embeddings: torch.Tensor,
        target_audio_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss with individual components.

        Args:
            za: Mapped audio embeddings in shared space
            zb: Mapped EEG embeddings in shared space
            predicted_audio_embeddings: Predicted audio embeddings (E_A_hat)
            target_audio_embeddings: Target audio embeddings (E_A)

        Returns:
            Tuple of:
                - total_loss: Combined weighted loss
                - loss_dict: Dictionary containing individual loss components
        """
        align_loss = self.alignment_loss(za, zb)
        rec_loss = self.reconstruction_loss(
            predicted_audio_embeddings,
            target_audio_embeddings
        )

        total_loss = self.lambda1 * align_loss + self.lambda2 * rec_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "alignment_loss": align_loss.item(),
            "reconstruction_loss": rec_loss.item()
        }

        return total_loss, loss_dict