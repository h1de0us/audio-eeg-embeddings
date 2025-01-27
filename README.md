## Overview

We take a pretrained encoder for audio data (encoder A) and train an encoder for EEG data (encoder B). Specifically I propose to use WavTokenizer as an encoder, because it produces a very dense (1 second -> 40 tokens) yet informative audio representation; also we can convert EEG data into wav format without any compression or data loss. Then we take the representations from encoder A and encoder B and use them as an input for the mapping layers, the mapping layers map the embeddings and into shared space. The mapping layers are trained on NMED dataset simultaneously with the predictor (predictor P takes EEG embedding from shared space as an input and produces audio embedding from encoder A's space as an output)

**Note**: I think it's strange to use a pretrained-on-audio encoder with EEG waves, though it may be an option in case we wouldn't be able to train our own encoder due to its computational complexity and costs ([Quote from the original paper](https://arxiv.org/pdf/2408.16532): *We train WavTokenizer-small up to 1 million iterations, with 500 thousand iterations allocated to both the generator and discriminator on 16 NVIDIA A800 80G*…)

## More about shared space

- let $E_A$  be the embeddings generated by encoder A, and $E_B$ be those from encoder B.
- we create mapping layers $M_A$ and $M_B$ for each encoder's output:
    - $M_A(E_A) \rightarrow Z_A$  maps $E_A$ to the shared space.
    - $M_B(E_B) \rightarrow Z_B$ maps  $E_B$ to the shared space.
- we want  $Z_A$   and $Z_B$  to be as similar as possible in the shared space, so we use a loss function that minimises the distance between  $Z_A$  and $Z_B$  ($\text{Loss}_{align}$).
- to predict  $E_A$   from  $E_B$ , (audio from EEG), we need an additional layer or network $P$, for instance, a GAN-like architecture
- $P(Z_B) \rightarrow \hat{E_A}$ , where  $\hat{E_A}$  is the predicted version of  $E_A$ ,
- then we train $P$ to minimise the difference between the predicted embedding  $\hat{E_A}$  and the actual embedding  $E_A$ :  $\text{Loss}_{rec} = \frac{1}{N} \sum_{i=1}^{N} ||\hat{E_A} - E_A||^2$  (**Note**: Loss_rec may be more complicated as in HifiGAN of similar models, here we can try different approaches)
- $\text{total loss} = \lambda_1 \cdot \text{Loss}_{align} + \lambda_2 \cdot \text{Loss}_{rec}$
- $M_A, M_B$  and the predictor  $P$ are trained simultaneously on [NMED dataset](https://exhibits.stanford.edu/data/catalog/jn859kj8079)

### Summary about mapping:

- input:  $E_A$   and  $E_B$
- output:  $Z_A = M_A(E_A) ,  Z_B = M_B(E_B)$ , and  $\hat{E_A} = P(Z_B)$
- loss:  $\lambda_1 \cdot \text{Loss}_{align} + \lambda_2 \cdot \text{Loss}_{rec}$

## Global summary

- pretrained encoder $A$ for audio
- train encoder $B$ for EEG on several EEG datasets (or take pretrained encoder $B$ == encoder $A$)
- take embeddings from $A$ and $B$ and map them into shared space using mapper $M$
- train a predictor $P$ on embeddings from shared space

## Project Structure

```
project_root/
├── configs/
│   ├── model_config.yaml        # Configuration for model architectures
│   └── training_config.yaml     # Training hyperparameters, loss weights (λ1, λ2)
│
├── data/
│   ├── raw/                     # Original NMED dataset
│   │   ├── audio/
│   │   └── eeg/
│   ├── processed/               # Preprocessed data
│   │   ├── audio_embeddings/    # Cached embeddings from encoder A
│   │   └── eeg_embeddings/      # Cached embeddings from encoder B
│   └── dataset.py              # Dataset classes and data loading utilities
│
├── models/
│   ├── encoders/
│   │   ├── audio_encoder.py     # WavTokenizer wrapper (encoder A)
│   │   └── eeg_encoder.py       # EEG encoder implementation (encoder B)
│   ├── mapping/
│   │   ├── mapping_layers.py    # Implementation of MA and MB mapping layers
│   │   └── predictor.py         # Predictor P implementation
│   └── loss.py                  # Loss functions (Loss_align, Loss_rec)
│
├── utils/
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── visualization.py         # Visualization tools for embeddings
│   └── metrics.py              # Evaluation metrics
│
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Citations

@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}
