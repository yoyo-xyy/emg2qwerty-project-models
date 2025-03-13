# EMG2QWERTY Project Models

This repository contains the critical files for the C147/247 Final Project (Winter 2025, Professor Jonathan Kao) at UCLA. The project builds upon the `emg2qwerty` dataset and codebase from Meta, focusing on surface electromyography (sEMG) decoding for touch typing on a QWERTY keyboard. We implement and compare three models—Transformer, LSTM, and hybrid GRU-CNN—against the baseline, achieving improved character error rates (CER) for single-subject decoding.

## Project Overview

The goal of this project is to decode sEMG signals from wristbands into QWERTY keystrokes, leveraging the `emg2qwerty` dataset (1,135 session files, 108 users, 346 hours). We developed three models:
- **LSTM**: Recorded a validation CER of 19.61 and test CER of 20.81, with better generalization.
- **Hybrid GRU-CNN**: Outperformed all models with a validation CER of 16.44 and test CER of 16.68, showing the best generalization.
- **Transformer**: Achieved a validation CER of 14.86 but overfit (test CER: 40.80) due to sensitivity to temporal downsampling.

Our work extends the original `emg2qwerty` repository by implementing new architectures, optimizing hyperparameters, and analyzing model performance for single-subject tasks. For detailed results, refer to the project report.

## Repository Structure

This repository includes the critical files modified or created for the project:

- `lightning.py`: Updated PyTorch Lightning module with implementations of LSTM, Hybrid GRU-CNN, and Transformer models.
- `modules.py`: Defines the model architectures, including the LSTM (4-layer bidirectional, 256 hidden units), GRU-CNN (2 CNN layers, 3-layer bidirectional GRU, 512 hidden units), and Transformer (2 layers, 8 heads, `TimePool`).
- `lstm/lstm_ctc.yaml`: Hyperparameters for the LSTM model.
- `hybrid-gru-cnn/hybrid_conv_gru.yaml`: Hyperparameters for the Hybrid GRU-CNN model.
- `transformer/transformer_ctc.yaml`: Hyperparameters for the Transformer model (50 epochs, warmup scheduler).

Python Notebooks for training each model:
- `lstm/Colab_lstm.ipynb`: notebook for the LSTM model.
- `hybrid-gru-cnn/Colab_LSTM.ipynb`: notebook for the Hybrid GRU-CNN model.
- `transformer/Colab_transformer.ipynb`: notebook for the Transformer model.
- `baseline/Colab_baseline.ipynb`: notebook for the baseline model.

## Setup

This project builds on the `emg2qwerty` codebase. Follow these steps to set up the environment and integrate our files:

**Clone the Original Repository**:
   ```bash
   git clone git@github.com:joe-lin-tech/emg2qwerty.git ~/emg2qwerty
   cd ~/emg2qwerty
