# Data Directory Structure

This directory contains the NMED dataset and processed data for the EEG-to-Audio mapping project.

## Directory Structure

```
data/
├── raw/                     # Original NMED dataset
│   ├── audio/              # Raw audio files
│   │   └── .gitkeep
│   └── eeg/               # Raw EEG recordings
│       └── .gitkeep
│
├── processed/              # Preprocessed data
│   ├── audio_embeddings/  # Cached embeddings from WavTokenizer
│   │   └── .gitkeep
│   ├── eeg_embeddings/    # Cached embeddings from EEG encoder
│   │   └── .gitkeep
│   └── splits/            # Train/val/test splits
│       └── .gitkeep
│
└── README.md              # This file
```

## Data Format

### Raw Data

1. Audio Files:
   - Format: WAV
   - Sample Rate: 24kHz
   - Channels: Mono
   - Duration: 1 second segments

2. EEG Recordings:
   - Format: Raw EEG converted to WAV
   - Sample Rate: 24kHz (resampled)
   - Channels: Mono
   - Duration: Aligned with audio segments

### Processed Data

1. Audio Embeddings:
   - Shape: (N, 40, 512) where N is number of samples
   - Format: PyTorch tensors (.pt files)
   - 40 tokens per second
   - 512-dimensional embeddings

2. EEG Embeddings:
   - Shape: (N, 40, 512)
   - Format: PyTorch tensors (.pt files)
   - Aligned with audio embeddings

## Dataset Source

The NMED dataset can be downloaded from:
https://exhibits.stanford.edu/data/catalog/jn859kj8079

## Data Processing

See `src/utils/preprocessing.py` for the data processing pipeline. 