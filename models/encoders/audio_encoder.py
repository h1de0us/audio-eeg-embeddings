"""
WavTokenizer wrapper to serve as the audio encoder (encoder A).
This wrapper adapts the WavTokenizer model to provide a consistent interface
with our EEG encoder while maintaining its powerful audio representation capabilities.

Expected Input Format:
---------------------
1. Audio Format:
   - Shape: (batch_size, channels, time_steps)
   - Channels: 1 (mono)
   - Sample Rate: 24kHz
   - Time Steps: Preferably 24000 (1 second)
   - Values: Normalized to [-1, 1]

2. Output Format:
   - Shape: (batch_size, sequence_length=40, embedding_dim=512)
   - 40 tokens per second (WavTokenizer's standard)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torchaudio
from pathlib import Path

from WavTokenizer.encoder.model import EncodecModel
from WavTokenizer.encoder.utils import convert_audio

class WavTokenizerWrapper(nn.Module):
    """
    Wrapper for WavTokenizer to use as audio encoder.
    Provides consistent interface with EEG encoder while leveraging
    WavTokenizer's powerful audio representation capabilities.
    """
    def __init__(
        self,
        pretrained: bool = True,
        sample_rate: int = 24000,
        embedding_dim: int = 512,
        sequence_length: int = 40,
        device: Optional[torch.device] = None
    ):
        """
        Initialize WavTokenizer wrapper.
        
        Args:
            pretrained: Whether to load pretrained weights
            sample_rate: Target sample rate (default: 24kHz)
            embedding_dim: Embedding dimension (default: 512)
            sequence_length: Number of tokens per second (default: 40)
            device: Device to load model on (default: None, uses available device)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize WavTokenizer model
        self.model = EncodecModel.encodec_model_24khz(pretrained=pretrained)
        self.model.set_target_bandwidth(24.0)  # Use highest quality
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Add projection layer if needed
        encoder_dim = self.model.encoder.dimension
        if encoder_dim != embedding_dim:
            self.proj = nn.Linear(encoder_dim, embedding_dim)
        else:
            self.proj = nn.Identity()

    def preprocess_audio(
        self, 
        audio: torch.Tensor,
        src_sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess audio to match WavTokenizer's requirements.
        
        Args:
            audio: Input audio tensor (batch_size, channels, time_steps)
            src_sample_rate: Source sample rate if different from target
            
        Returns:
            Preprocessed audio tensor
        """
        if src_sample_rate and src_sample_rate != self.sample_rate:
            audio = convert_audio(audio, src_sample_rate, self.sample_rate, 1)
        
        # Ensure mono
        if audio.shape[1] > 1:
            audio = torch.mean(audio, dim=1, keepdim=True)
            
        return audio

    @torch.no_grad()
    def forward(
        self, 
        x: torch.Tensor,
        src_sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract audio embeddings using WavTokenizer.
        
        Args:
            x: Input audio tensor (batch_size, channels, time_steps)
            src_sample_rate: Source sample rate if different from target
            
        Returns:
            Audio embeddings tensor (batch_size, sequence_length, embedding_dim)
            
        Example:
            >>> encoder = WavTokenizerWrapper()
            >>> audio = torch.randn(32, 1, 24000)  # 1 second at 24kHz
            >>> embeddings = encoder(audio)  # Shape: (32, 40, 512)
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, channels, time_steps), "
                f"got shape {x.shape}"
            )
        
        # Preprocess audio
        x = self.preprocess_audio(x, src_sample_rate)
        x = x.to(self.device)
        
        # Extract embeddings using WavTokenizer
        emb = self.model.encoder(x)
        
        # Project to target dimension if needed
        emb = self.proj(emb.transpose(1, 2))  # Shape: (batch, time, dim)
        
        return emb

    def get_tokens(
        self,
        x: torch.Tensor,
        src_sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get discrete tokens from WavTokenizer (useful for debugging or analysis).
        
        Args:
            x: Input audio tensor
            src_sample_rate: Source sample rate if different from target
            
        Returns:
            Discrete tokens tensor
        """
        x = self.preprocess_audio(x, src_sample_rate)
        x = x.to(self.device)
        _, tokens = self.model.encode_infer(x, bandwidth_id=torch.tensor([0]))
        return tokens

def create_audio_encoder(config: dict) -> nn.Module:
    """
    Factory function to create audio encoder from config.
    
    Args:
        config: Dictionary containing model configuration
            Required keys:
                - pretrained: bool
                - sample_rate: int
                - embedding_dim: int
                - sequence_length: int
            Optional keys:
                - device: torch.device
    
    Returns:
        Configured WavTokenizer wrapper
    """
    return WavTokenizerWrapper(
        pretrained=config.get('pretrained', True),
        sample_rate=config.get('sample_rate', 24000),
        embedding_dim=config.get('embedding_dim', 512),
        sequence_length=config.get('sequence_length', 40),
        device=config.get('device', None)
    ) 