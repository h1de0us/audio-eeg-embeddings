"""
EEG Encoder for processing EEG data in WAV format.

Expected Input Format:
---------------------
The encoder expects EEG data to be preprocessed into WAV format with the following specifications:

1. Data Format:
   - Shape: (batch_size, channels, time_steps)
   - Channels: Typically 1 (mono WAV format)
   - Sample Rate: 16kHz (same as WavTokenizer)
   - Time Steps: Variable length, will be adaptively pooled to sequence_length

2. Preprocessing Requirements:
   - EEG data should be normalized to [-1, 1] range (standard WAV format)
   - Each sample should represent 1 second of EEG data
   - No compression or data loss during WAV conversion
   - Sampling frequency should match the target audio encoder

3. Example Preprocessing:
   raw_eeg = load_eeg_data()  # Your EEG data loading
   normalized_eeg = (raw_eeg - raw_eeg.mean()) / raw_eeg.std()  # Normalize
   wav_eeg = normalize_to_wav_range(normalized_eeg)  # Scale to [-1, 1]
   
4. Batch Example:
   - Input shape for 32 samples: (32, 1, 16000)  # 1 channel, 16kHz for 1 second
   - Output shape: (32, 40, 512)  # 40 tokens with 512 dimensions each

Note: The encoder uses adaptive pooling, so input length can vary, but 
      1 second of data at 16kHz is recommended for compatibility with WavTokenizer.
"""

import torch
import torch.nn as nn
from typing import Optional

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class EEGEncoder(nn.Module):
    """
    EEG encoder that processes EEG data in wav format and produces embeddings.
    Architecture inspired by WavTokenizer but simplified for EEG data.
    
    Input Requirements:
    ------------------
    - Format: WAV-formatted EEG data
    - Shape: (batch_size, channels, time_steps)
    - Value Range: [-1, 1]
    - Sample Rate: 16kHz (16000 samples per second)
    - Duration: Preferably 1 second per sample
    
    Output Format:
    -------------
    - Shape: (batch_size, sequence_length, embedding_dim)
    - sequence_length: 40 tokens (matching WavTokenizer)
    - embedding_dim: Configurable, default 512
    
    Example:
    --------
    >>> encoder = EEGEncoder(in_channels=1, embedding_dim=512)
    >>> x = torch.randn(32, 1, 16000)  # 32 samples, 1 channel, 1 second at 16kHz
    >>> embeddings = encoder(x)  # Shape: (32, 40, 512)
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_sizes: list[int] = [3, 3, 3, 3, 3],
        strides: list[int] = [2, 2, 2, 2, 2],
        dilations: list[int] = [1, 1, 1, 1, 1],
        embedding_dim: int = 512,
        sequence_length: int = 40  # Same as WavTokenizer (1 second -> 40 tokens)
    ):
        """
        Initialize the EEG encoder.
        
        Args:
            in_channels: Number of input channels in EEG data (default: 1 for WAV format)
            base_channels: Base number of channels for conv layers (default: 32)
            kernel_sizes: Kernel sizes for each conv layer (default: [3,3,3,3,3])
            strides: Strides for each conv layer (default: [2,2,2,2,2])
            dilations: Dilation rates for each conv layer (default: [1,1,1,1,1])
            embedding_dim: Dimension of output embeddings (default: 512)
            sequence_length: Number of output tokens (default: 40)
            
        Note:
            The default configuration expects input data sampled at 16kHz.
            The convolutional layers progressively reduce the temporal dimension
            while increasing the channel dimension, before adaptive pooling to
            the target sequence length.
        """
        super().__init__()
        
        assert len(kernel_sizes) == len(strides) == len(dilations), \
            "kernel_sizes, strides, and dilations must have the same length"
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        current_channels = in_channels
        
        for i, (kernel_size, stride, dilation) in enumerate(
            zip(kernel_sizes, strides, dilations)
        ):
            out_channels = base_channels * (2 ** i)
            self.encoder_layers.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation
                )
            )
            current_channels = out_channels
        
        # Projection to embedding dimension
        self.proj = nn.Sequential(
            nn.Conv1d(current_channels, embedding_dim, 1),
            nn.GELU()
        )
        
        # Adaptive pooling to get desired sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(sequence_length)
        
        # Layer normalization for final embeddings
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
               - channels: Typically 1 for WAV format
               - time_steps: Typically 16000 for 1 second at 16kHz
               - Values should be normalized to [-1, 1] range
                
        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
            - sequence_length: 40 tokens
            - embedding_dim: As specified in initialization
            
        Raises:
            ValueError: If input tensor doesn't match expected shape
            
        Example:
            >>> # For 1 second of EEG data at 16kHz
            >>> x = torch.randn(32, 1, 16000)  # (batch_size, channels, time_steps)
            >>> output = encoder(x)  # Shape: (32, 40, 512)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, channels, time_steps), "
                f"got shape {x.shape}"
            )
        
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )
            
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Project to embedding dimension
        x = self.proj(x)  # Shape: (batch, embedding_dim, time)
        
        # Pool to desired sequence length
        x = self.adaptive_pool(x)  # Shape: (batch, embedding_dim, sequence_length)
        
        # Transpose and normalize
        x = x.transpose(1, 2)  # Shape: (batch, sequence_length, embedding_dim)
        x = self.layer_norm(x)
        
        return x

class EEGEncoderWithPositionalEncoding(EEGEncoder):
    """EEG encoder with additional positional encoding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, kwargs.get('sequence_length', 40), kwargs.get('embedding_dim', 512))
        )
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x + self.pos_encoding

def create_eeg_encoder(config: dict) -> nn.Module:
    """
    Factory function to create EEG encoder based on configuration.
    
    Args:
        config: Dictionary containing model configuration
            Required keys:
                - use_positional_encoding: bool
                - in_channels: int
                - base_channels: int
                - embedding_dim: int
                - sequence_length: int
            Optional keys:
                - kernel_sizes: list[int]
                - strides: list[int]
                - dilations: list[int]
    
    Returns:
        Configured EEG encoder model
    """
    encoder_class = (
        EEGEncoderWithPositionalEncoding 
        if config.get('use_positional_encoding', False)
        else EEGEncoder
    )
    
    return encoder_class(
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        embedding_dim=config['embedding_dim'],
        sequence_length=config['sequence_length'],
        kernel_sizes=config.get('kernel_sizes', [3, 3, 3, 3, 3]),
        strides=config.get('strides', [2, 2, 2, 2, 2]),
        dilations=config.get('dilations', [1, 1, 1, 1, 1])
    ) 