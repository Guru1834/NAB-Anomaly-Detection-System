#  NAB Anomaly Detection System
### GRU Autoencoder for Real-Time Network Anomaly Detection

A deep learning pipeline that detects anomalies in AWS cloud infrastructure time-series data using a **Gated Recurrent Unit (GRU) Autoencoder**. Trained on the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) `realAWSCloudwatch` dataset, the model learns what "normal" looks like — and flags anything that deviates.

---

##  Dataset

**Source:** NAB — `realAWSCloudwatch`

Three real AWS infrastructure metrics, each sampled at 5-minute intervals:

| Series | File | What it measures |
|---|---|---|
| EC2 Network In | `ec2_network_in_5abac7.csv` | Inbound network bytes to an EC2 instance |
| EC2 CPU Utilization | `ec2_cpu_utilization_53ea38.csv` | CPU usage % of an EC2 instance |
| ELB Request Count | `elb_request_count_8c0756.csv` | Requests hitting an Elastic Load Balancer |

A separate GRU Autoencoder is trained and evaluated on each series independently.

---

## Architecture

```
Raw Time-Series
      │
      ▼
┌─────────────────┐
│  MinMaxScaler   │  → Normalize values to [0, 1]
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Sliding Windows │  → seq_len = 30 steps per window
└─────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│            GRU AUTOENCODER               │
│                                          │
│  Input Window (30 × 1)                   │
│         │                                │
│         ▼                                │
│  ┌─────────────┐                         │
│  │ Encoder GRU │  2 layers, hidden=64    │
│  └─────────────┘                         │
│         │  hidden state h_n              │
│         ▼                                │
│  ┌────────────┐                          │
│  │ Bottleneck │  Linear + ReLU           │
│  └────────────┘                          │
│         │  context vector                │
│         ▼                                │
│  ┌─────────────┐                         │
│  │ Decoder GRU │  2 layers, hidden=64    │
│  └─────────────┘                         │
│         │                                │
│         ▼                                │
│  ┌──────────────┐                        │
│  │ Output Layer │  Linear → (30 × 1)     │
│  └──────────────┘                        │
└──────────────────────────────────────────┘
      │
      ▼
Reconstruction Error (MSE per window)
      │
      ▼
Threshold = μ + 2.5σ  (computed on training errors only)
      │
      ▼
Anomaly if Error > Threshold
```

**Key design choice:** The model is trained **only on the first 70% of each series** (assumed normal behavior). It learns to reconstruct normal patterns well — so when something unusual happens in the remaining 30%, reconstruction error spikes and triggers a flag.

---

## Pipeline — Step by Step

### Step 1 — Load & Parse
Load each CSV, parse `timestamp` to datetime, and verify shapes and missing values.

### Step 2 — Exploratory Data Analysis
- Descriptive statistics (`describe()`, `info()`)
- Hourly resampled time-series plots for each metric
- Missing value audit across all three series

### Step 3 — Feature Engineering
Temporal features extracted from each timestamp:
```
year | month | day | hour | minute | day_of_week
```

### Step 4 — Sequence Preprocessing
```python
# Normalize to [0, 1]
scaler = MinMaxScaler()
norm = scaler.fit_transform(values)

# Sliding window (seq_len = 30)
# Shape: (N - seq_len + 1, seq_len, 1)
sequences = [series[i : i + seq_len] for i in range(len(series) - seq_len + 1)]

# 70/30 train-test split
X_train = sequences[:int(0.70 * len(sequences))]
X_all   = sequences  # used for scoring
```

### Step 5 — Model Training
```
Optimizer  : Adam  (lr=1e-3, weight_decay=1e-5)
Loss       : MSELoss (reconstruction)
Epochs     : up to 60
Batch size : 64
Grad clip  : max_norm = 1.0
LR Scheduler : ReduceLROnPlateau (patience=3, factor=0.5)
Early stopping : patience = 7 epochs
```

The best model checkpoint (lowest training loss) is restored after early stopping.

### Step 6 — Anomaly Scoring
```python
# Per-window reconstruction error
error = mean((reconstructed - original) ** 2)

# Threshold derived from training data only
threshold = mean(train_errors) + 2.5 * std(train_errors)

# Flag
anomaly = 1 if error > threshold else 0
```

### Step 7 — Evaluation & Visualisation
Each series produces a **4-panel result figure**:
- Panel 1 — Raw time-series with train/test split marker
- Panel 2 — Reconstruction error curve with threshold line
- Panel 3 — Detected anomalies overlaid on raw series (red dots)
- Panel 4 — Training loss curve per epoch

A **cross-metric comparison plot** (`gru_cross_metric_comparison.png`) shows all three series side by side for a unified view.

### Step 8 — Save Outputs
```
gru_outputs/
├── gru_network_in.pth        ← trained model weights
├── gru_cpu.pth
├── gru_elb_requests.pth
├── results_network_in.csv    ← timestamp | value | recon_error | anomaly_gru
├── results_cpu.csv
└── results_elb_requests.csv
```

---

## ⚙️ Model Hyperparameters

| Parameter | Value |
|---|---|
| Sequence length | 30 |
| Hidden dimension | 64 |
| GRU layers | 2 |
| Dropout | 0.2 |
| Bottleneck activation | ReLU |
| Anomaly threshold | μ + 2.5σ |
| Train fraction | 70% |
| Total parameters | ~100K |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | GRU Autoencoder, training loop |
| Scikit-learn | MinMaxScaler, Precision / Recall / F1 |
| Pandas / NumPy | Data loading, sliding windows, feature engineering |
| Matplotlib / Seaborn | Time-series and anomaly visualisations |

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Guru1834/NAB-Anomaly-Detection-System.git
cd NAB-Anomaly-Detection-System

# Install dependencies
pip install torch scikit-learn pandas numpy matplotlib seaborn

# Download the NAB dataset
# https://github.com/numenta/NAB
# Place CSVs under: data/realAWSCloudwatch/

# Run the notebook
jupyter notebook E2C_Activity_GRU.ipynb
```

---

## Project Structure

```
NAB-Anomaly-Detection-System/
│
├── E2C_Activity_GRU.ipynb          ← Main notebook (EDA + model + results)
├── README.md
│
├── data/
│   └── realAWSCloudwatch/
│       ├── ec2_network_in_5abac7.csv
│       ├── ec2_cpu_utilization_53ea38.csv
│       └── elb_request_count_8c0756.csv
│
└── gru_outputs/                    ← Generated after running the notebook
    ├── *.pth                       ← Model weights
    ├── results_*.csv               ← Per-point anomaly labels
    └── gru_cross_metric_comparison.png
```

---
