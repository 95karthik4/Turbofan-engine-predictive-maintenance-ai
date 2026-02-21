# ✈️ Turbofan Engine Predictive Maintenance AI

> A deep learning system to predict the **Remaining Useful Life (RUL)** of jet engines using LSTMs and Transformers, with a live **Streamlit** monitoring dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Dashboard](#dashboard)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Predictive maintenance is a critical application of AI in aerospace and industrial settings. This project uses the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset to train deep learning models that estimate how many operational cycles remain before a turbofan engine requires maintenance or fails.

The system combines:
- **LSTM (Long Short-Term Memory)** networks to capture temporal degradation patterns in sensor time series data.
- **Transformer** models to leverage attention mechanisms for long-range dependency modelling across sensor readings.
- A **Streamlit** dashboard for real-time engine health monitoring and RUL visualisation.

---

## Features

- 🔬 **Multi-model support** — train and compare LSTM and Transformer architectures side-by-side.
- 📊 **Sensor fusion** — processes 21 raw sensor signals and 3 operational settings simultaneously.
- 🔄 **Sliding window preprocessing** — converts raw time series into fixed-length input sequences for deep learning.
- 📈 **Live Streamlit dashboard** — visualise sensor trends, predicted RUL, and model confidence in real time.
- 🗄️ **Multi-dataset support** — compatible with all four CMAPSS sub-datasets (FD001–FD004).
- 📉 **Custom loss function** — asymmetric loss penalises late predictions more heavily than early ones, matching real-world maintenance requirements.
- 💾 **Model checkpointing** — automatically saves the best model weights during training.

---

## Dataset

This project uses the **NASA CMAPSS Turbofan Engine Degradation Simulation** dataset.

| Sub-dataset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|-------------|:-------------:|:------------:|:--------------------:|:-----------:|
| FD001        | 100           | 100          | 1                    | 1           |
| FD002        | 260           | 259          | 6                    | 1           |
| FD003        | 100           | 100          | 1                    | 2           |
| FD004        | 248           | 248          | 6                    | 2           |

Each engine run contains:
- **3 operational settings** (altitude, throttle, Mach number)
- **21 sensor measurements** (temperatures, pressures, fan speeds, fuel flows, etc.)
- A ground-truth RUL value for each test engine

**Download the dataset:**

1. Visit the [NASA Prognostics Center of Excellence Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6).
2. Download and unzip the dataset into the `data/` directory:

```
data/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

---

## Model Architecture

### LSTM Model
A stacked bidirectional LSTM network that processes windowed sensor sequences to output a scalar RUL estimate. Dropout regularisation is applied between layers to prevent overfitting.

```
Input (window × sensors) → BiLSTM → Dropout → BiLSTM → Dropout → FC → ReLU → FC → RUL
```

### Transformer Model
A Transformer encoder with multi-head self-attention is used to capture complex, long-range dependencies between sensor readings across the time window. Positional encoding is added to retain temporal ordering.

```
Input (window × sensors) → Positional Encoding → Transformer Encoder (N layers) → Global Avg Pool → FC → ReLU → FC → RUL
```

---

## Project Structure

```
Turbofan-engine-predictive-maintenance-ai/
│
├── data/                        # Raw CMAPSS dataset files (not tracked by git)
│
├── models/                      # Model definitions
│   ├── lstm_model.py            # Bidirectional LSTM architecture
│   └── transformer_model.py     # Transformer encoder architecture
│
├── utils/                       # Data loading and preprocessing utilities
│   ├── data_loader.py           # Dataset class and DataLoader helpers
│   └── preprocessing.py         # Normalisation, windowing, and RUL clipping
│
├── training/
│   ├── train.py                 # Training loop and checkpoint saving
│   └── evaluate.py              # RMSE, MAE, and NASA scoring function
│
├── dashboard/
│   └── app.py                   # Streamlit monitoring dashboard
│
├── checkpoints/                 # Saved model weights (not tracked by git)
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA and feature engineering notebook
│
├── requirements.txt             # Python dependencies
├── config.yaml                  # Hyperparameter and training configuration
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Steps

1. **Clone the repository:**

```bash
git clone https://github.com/95karthik4/Turbofan-engine-predictive-maintenance-ai.git
cd Turbofan-engine-predictive-maintenance-ai
```

2. **Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download and place the CMAPSS dataset** into the `data/` directory as described in the [Dataset](#dataset) section.

---

## Usage

### Training

Train a model using the configuration in `config.yaml`:

```bash
python training/train.py --dataset FD001 --model lstm --epochs 100
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | CMAPSS sub-dataset (`FD001`–`FD004`) | `FD001` |
| `--model` | Model type (`lstm` or `transformer`) | `lstm` |
| `--epochs` | Number of training epochs | `100` |
| `--batch_size` | Mini-batch size | `64` |
| `--lr` | Learning rate | `0.001` |
| `--window` | Sliding window length (cycles) | `30` |
| `--max_rul` | RUL clipping threshold | `125` |

Best model weights are saved to `checkpoints/`.

---

### Evaluation

Evaluate a trained model on the test split:

```bash
python training/evaluate.py --dataset FD001 --model lstm --checkpoint checkpoints/best_lstm_FD001.pth
```

Reported metrics:
- **RMSE** — Root Mean Square Error
- **MAE** — Mean Absolute Error
- **NASA Score** — Asymmetric scoring function that penalises late predictions more heavily

---

### Dashboard

Launch the Streamlit monitoring dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Real-time sensor trend visualisation per engine
- Predicted RUL with confidence intervals
- Fleet-level health overview
- Model comparison (LSTM vs Transformer)

Navigate to `http://localhost:8501` in your browser.

---

## Results

Benchmark results on the CMAPSS test sets (lower RMSE is better):

| Model       | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|-------------|:----------:|:----------:|:----------:|:----------:|
| LSTM        | ~14.5      | ~22.3      | ~15.2      | ~23.8      |
| Transformer | ~13.1      | ~20.7      | ~13.9      | ~21.4      |

> Results are indicative; exact values depend on hyperparameter tuning.

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

Please ensure your code follows the existing style and include relevant tests where applicable.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division) for the CMAPSS dataset.
- The PyTorch and Streamlit communities for their excellent open-source tooling.
