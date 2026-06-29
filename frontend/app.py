"""
NAB Anomaly Detection Dashboard
Streamlit app — upload any NAB CloudWatch CSV and run anomaly detection.
"""

import io
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAB Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #8b949e !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .val {
    font-size: 28px;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.2;
}
.metric-card .lbl {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.metric-card.red .val  { color: #f85149; }
.metric-card.blue .val { color: #58a6ff; }
.metric-card.grn .val  { color: #3fb950; }

/* Section header */
.sec-header {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* Method badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 8px;
}
.badge-gru   { background: #1f2d3d; color: #58a6ff; }
.badge-stat  { background: #1d2a1d; color: #3fb950; }
.badge-ml    { background: #2d1f2d; color: #bc8cff; }
.badge-ens   { background: #2d2214; color: #e3b341; }

/* Run button */
.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-top: 8px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #2ea043 !important;
}

/* Tab style */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #8b949e;
    font-size: 13px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* Plotly chart borders */
.element-container iframe,
.js-plotly-plot { border-radius: 10px; }

/* Sidebar divider */
.sidebar-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1rem 0;
}

/* Info box */
.info-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #8b949e;
    line-height: 1.6;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# GRU AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────
class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features=1, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_gru = nn.GRU(
            n_features, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.bottleneck = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_gru = nn.GRU(
            hidden_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, h_n = self.encoder_gru(x)
        context = torch.relu(self.bottleneck(h_n[-1]))
        dec_input = context.unsqueeze(1).expand(-1, self.seq_len, -1)
        decoded, _ = self.decoder_gru(dec_input)
        return self.output_layer(decoded)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION METHODS
# ─────────────────────────────────────────────────────────────────────────────
def detect_zscore(values, threshold=3.0, window=30):
    s = pd.Series(values)
    rm = s.rolling(window, min_periods=1).mean()
    rs = s.rolling(window, min_periods=1).std().fillna(0)
    z = (s - rm) / (rs + 1e-8)
    return (np.abs(z) > threshold).astype(int).values, np.abs(z).values


def detect_iqr(values, factor=1.75):
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    flags = ((values < lo) | (values > hi)).astype(int)
    score = np.maximum(lo - values, values - hi) / (iqr + 1e-8)
    return flags, np.clip(score, 0, None)


def detect_isoforest(values, contamination=0.05):
    s = pd.Series(values)
    X = pd.DataFrame({
        "v": values,
        "rm": s.rolling(10, min_periods=1).mean(),
        "rs": s.rolling(10, min_periods=1).std().fillna(0),
        "d1": s.diff(1).fillna(0),
    }).values
    X = StandardScaler().fit_transform(X)
    clf = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)
    scores = -clf.decision_function(X)
    mn, mx = scores.min(), scores.max()
    scores = (scores - mn) / (mx - mn + 1e-8)
    return (preds == -1).astype(int), scores


def detect_lof(values, contamination=0.05, n_neighbors=20):
    s = pd.Series(values)
    X = pd.DataFrame({
        "v": values,
        "rm": s.rolling(10, min_periods=1).mean(),
        "rs": s.rolling(10, min_periods=1).std().fillna(0),
        "d1": s.diff(1).fillna(0),
    }).values
    X = StandardScaler().fit_transform(X)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = clf.fit_predict(X)
    scores = -clf.negative_outlier_factor_
    mn, mx = scores.min(), scores.max()
    scores = (scores - mn) / (mx - mn + 1e-8)
    return (preds == -1).astype(int), scores


def make_sequences(series, seq_len):
    return np.array(
        [series[i:i + seq_len] for i in range(len(series) - seq_len + 1)],
        dtype=np.float32
    )[..., np.newaxis]


@torch.no_grad()
def recon_errors(model, X_all, batch_size=256):
    model.eval().to(DEVICE)
    out = []
    for i in range(0, len(X_all), batch_size):
        xb = X_all[i:i + batch_size].to(DEVICE)
        err = ((model(xb) - xb) ** 2).mean(dim=(1, 2))
        out.append(err.cpu().numpy())
    return np.concatenate(out)


def train_gru(values, seq_len=30, epochs=50, lr=1e-3, sigma=2.5,
              train_frac=0.7, progress_cb=None):
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    n_train = int(len(norm) * train_frac)

    train_seq = make_sequences(norm[:n_train], seq_len)
    all_seq   = make_sequences(norm, seq_len)
    X_train   = torch.tensor(train_seq, dtype=torch.float32)
    X_all     = torch.tensor(all_seq,   dtype=torch.float32)

    model = GRUAutoencoder(seq_len).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit  = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)

    best_loss, stale, best_state = np.inf, 0, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_train)
        history.append(ep_loss)
        sched.step(ep_loss)
        if progress_cb:
            progress_cb(epoch, epochs, ep_loss)
        if ep_loss < best_loss - 1e-6:
            best_loss, stale = ep_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 7:
                break

    model.load_state_dict(best_state)
    model.eval()

    errors = recon_errors(model, X_all)
    train_err = errors[:len(X_train)]
    threshold = train_err.mean() + sigma * train_err.std()
    labels = (errors > threshold).astype(int)

    # pad front so length matches original series
    offset = seq_len - 1
    full_labels = np.zeros(len(values), dtype=int)
    full_scores = np.zeros(len(values))
    full_labels[offset:] = labels
    full_scores[offset:] = (errors - threshold) / (threshold + 1e-8)

    return full_labels, np.clip(full_scores, 0, None), history, n_train, offset, threshold


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG   = "#0d1117"
PLOT_PAPER = "#0d1117"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#c9d1d9"
BLUE      = "#58a6ff"
RED       = "#f85149"
PURPLE    = "#bc8cff"
GREEN     = "#3fb950"
ORANGE    = "#e3b341"


def base_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_CLR, size=14)),
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, size=11),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR, borderwidth=1),
        hovermode="x unified",
    )


def plot_raw(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["value"],
        mode="lines", name="value",
        line=dict(color=BLUE, width=1.2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"
    ))
    fig.update_layout(**base_layout("Raw time series", 320))
    return fig


def plot_results(df, labels, scores, method, n_train=None, offset=None):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )
    ts, vals = df["timestamp"], df["value"]

    # Panel 1 — raw + anomalies
    fig.add_trace(go.Scatter(
        x=ts, y=vals, mode="lines", name="value",
        line=dict(color=BLUE, width=1.0),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"
    ), row=1, col=1)

    anom_mask = labels == 1
    if anom_mask.any():
        fig.add_trace(go.Scatter(
            x=ts[anom_mask], y=vals[anom_mask],
            mode="markers", name=f"anomaly ({anom_mask.sum()})",
            marker=dict(color=RED, size=6, symbol="circle",
                        line=dict(color="#ff7b72", width=1)),
        ), row=1, col=1)

    # Train/test split line
    if n_train is not None and n_train < len(ts):
        split_ts = ts.iloc[n_train]
        fig.add_vline(x=split_ts, line_dash="dash",
                      line_color="#8b949e", line_width=1,
                      annotation_text="train | test",
                      annotation_font_color="#8b949e",
                      annotation_font_size=10)

    # Panel 2 — anomaly score
    score_ts = ts.iloc[offset:] if offset else ts
    score_vals = scores[offset:] if offset else scores
    fig.add_trace(go.Scatter(
        x=score_ts, y=score_vals,
        mode="lines", name="anomaly score",
        line=dict(color=PURPLE, width=0.9),
        fill="tozeroy", fillcolor="rgba(188,140,255,0.1)"
    ), row=2, col=1)

    # Threshold line at 0 (normalised) or 1.0
    fig.add_hline(
        y=0, line_dash="dash", line_color=RED, line_width=1.0,
        annotation_text="threshold", annotation_font_color=RED,
        annotation_font_size=9, row=2, col=1
    )

    layout = base_layout(f"Anomaly detection — {method}", 480)
    layout["showlegend"] = True
    layout["xaxis2"] = dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False)
    layout["yaxis2"] = dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
                             title="score")
    fig.update_layout(**layout)
    return fig


def plot_training_loss(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history, mode="lines+markers", name="MSE loss",
        line=dict(color=GREEN, width=1.5),
        marker=dict(size=4, color=GREEN)
    ))
    fig.update_layout(**base_layout("GRU training loss", 260))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="MSE")
    return fig


def plot_comparison(df, results):
    methods = list(results.keys())
    n = len(methods)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=methods)
    colors = [BLUE, GREEN, PURPLE, ORANGE, RED, "#ffa657"]
    ts, vals = df["timestamp"], df["value"]

    for i, (mname, (labels, _)) in enumerate(results.items()):
        row = i + 1
        col = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=ts, y=vals, mode="lines", name="value",
            line=dict(color="#4d5566", width=0.8),
            showlegend=(i == 0)
        ), row=row, col=1)
        anom = labels == 1
        if anom.any():
            fig.add_trace(go.Scatter(
                x=ts[anom], y=vals[anom],
                mode="markers", name=f"{mname} ({anom.sum()})",
                marker=dict(color=RED, size=5)
            ), row=row, col=1)

    layout = base_layout("Method comparison", height=230 * n)
    layout["showlegend"] = True
    for k in list(layout.keys()):
        if k.startswith("xaxis") or k.startswith("yaxis"):
            layout.pop(k)
    layout["paper_bgcolor"] = PLOT_PAPER
    layout["plot_bgcolor"]  = PLOT_BG
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID_CLR, showgrid=True)
    fig.update_yaxes(gridcolor=GRID_CLR, showgrid=True)
    return fig


def plot_score_distribution(scores, threshold_idx=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=60, name="score distribution",
        marker_color=PURPLE, opacity=0.75
    ))
    fig.update_layout(**base_layout("Anomaly score distribution", 260))
    fig.update_xaxes(title_text="score")
    fig.update_yaxes(title_text="count")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 NAB Anomaly\nDetection")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Data source
    st.markdown("**Data source**")
    upload_mode = st.radio(
        "", ["Upload CSV", "Demo (synthetic)"],
        label_visibility="collapsed"
    )

    df_raw = None

    if upload_mode == "Upload CSV":
        uploaded = st.file_uploader(
            "NAB CloudWatch CSV",
            type=["csv"],
            help="CSV must have a 'value' column and optionally a 'timestamp' column."
        )
        if uploaded:
            try:
                df_raw = pd.read_csv(uploaded)
                if "timestamp" in df_raw.columns:
                    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                else:
                    df_raw["timestamp"] = pd.date_range(
                        "2024-01-01", periods=len(df_raw), freq="5min"
                    )
                if "value" not in df_raw.columns:
                    st.error("CSV must contain a 'value' column.")
                    df_raw = None
            except Exception as e:
                st.error(f"Could not read file: {e}")

    else:
        # Generate synthetic demo data with injected anomalies
        st.markdown('<div class="info-box">Demo uses synthetic time-series data with 8 injected anomaly spikes.</div>', unsafe_allow_html=True)
        n_pts = 2000
        t = np.arange(n_pts)
        np.random.seed(42)
        vals = (
            50 + 10 * np.sin(2 * np.pi * t / 288)
            + 5 * np.sin(2 * np.pi * t / 48)
            + np.random.normal(0, 2, n_pts)
        )
        spike_idxs = [350, 600, 850, 1050, 1300, 1550, 1700, 1900]
        for idx in spike_idxs:
            vals[idx:idx+3] += np.random.uniform(30, 55)
        df_raw = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_pts, freq="5min"),
            "value": vals,
        })

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Method
    st.markdown("**Detection method**")
    METHOD_OPTIONS = [
        "GRU Autoencoder",
        "Z-Score",
        "IQR",
        "Isolation Forest",
        "Local Outlier Factor",
        "All Methods (ensemble)",
    ]
    method = st.selectbox("", METHOD_OPTIONS, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Method-specific params
    st.markdown("**Parameters**")

    if method in ("GRU Autoencoder", "All Methods (ensemble)"):
        seq_len   = st.slider("Sequence length", 10, 60, 30)
        epochs    = st.slider("Max epochs", 20, 150, 50)
        sigma     = st.slider("Threshold σ", 1.5, 4.0, 2.5, 0.1)
        train_frac = st.slider("Train fraction", 0.5, 0.9, 0.7, 0.05)

    if method in ("Z-Score",):
        z_thresh  = st.slider("Z threshold", 1.5, 5.0, 3.0, 0.1)
        z_window  = st.slider("Rolling window", 5, 60, 30)

    if method in ("IQR",):
        iqr_factor = st.slider("IQR multiplier", 1.0, 3.0, 1.75, 0.05)

    if method in ("Isolation Forest", "Local Outlier Factor",
                  "All Methods (ensemble)"):
        contam = st.slider("Contamination", 0.01, 0.15, 0.05, 0.01)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Detection")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#484f58">Built with PyTorch · scikit-learn · Plotly<br>NAB Dataset — Numenta</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# NAB Anomaly Detection Dashboard")
st.markdown('<div style="color:#8b949e;font-size:13px;margin-bottom:1rem">Upload a NAB CloudWatch CSV, pick a method, hit Run.</div>', unsafe_allow_html=True)

if df_raw is None and upload_mode == "Upload CSV":
    st.markdown("""
    <div class="info-box">
    ⬆️ Upload a NAB CSV file from the sidebar to get started.<br><br>
    Expected format: <code>timestamp, value</code> — any AWS CloudWatch file from the NAB dataset works.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = df_raw.copy().reset_index(drop=True)
values = df["value"].values.astype(float)
n = len(df)

# Dataset stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{n:,}")
with col2:
    st.metric("Min value", f"{values.min():.2f}")
with col3:
    st.metric("Max value", f"{values.max():.2f}")
with col4:
    st.metric("Mean value", f"{values.mean():.2f}")

tab_overview, tab_results, tab_compare, tab_export = st.tabs(
    ["📈 Overview", "🔴 Detection results", "⚖️ Method comparison", "💾 Export"]
)

# ── TAB 1: OVERVIEW ─────────────────────────────────────────────────────────
with tab_overview:
    st.plotly_chart(plot_raw(df), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Descriptive statistics**")
        st.dataframe(
            df["value"].describe().rename("value").to_frame().style.format("{:.4f}"),
            use_container_width=True
        )
    with col_b:
        st.markdown("**Sample rows**")
        st.dataframe(df.head(10), use_container_width=True)


# ── RUN DETECTION ────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}
if "gru_history" not in st.session_state:
    st.session_state.gru_history = None
if "last_method" not in st.session_state:
    st.session_state.last_method = None

if run_btn:
    st.session_state.results = {}
    st.session_state.gru_history = None

    with tab_results:
        methods_to_run = (
            ["GRU Autoencoder", "Z-Score", "IQR", "Isolation Forest", "Local Outlier Factor"]
            if method == "All Methods (ensemble)"
            else [method]
        )

        for m in methods_to_run:
            status = st.empty()
            status.info(f"Running {m}…")

            if m == "GRU Autoencoder":
                prog_bar  = st.progress(0)
                prog_text = st.empty()

                def gru_cb(ep, total, loss):
                    prog_bar.progress(ep / total)
                    prog_text.markdown(
                        f'<div style="font-size:12px;color:#8b949e">Epoch {ep}/{total} · loss {loss:.6f}</div>',
                        unsafe_allow_html=True
                    )

                labels, scores, history, n_train, offset, thresh = train_gru(
                    values, seq_len=seq_len, epochs=epochs, sigma=sigma,
                    train_frac=train_frac, progress_cb=gru_cb
                )
                st.session_state.gru_history = history
                st.session_state.gru_n_train = n_train
                st.session_state.gru_offset  = offset
                prog_bar.empty()
                prog_text.empty()

            elif m == "Z-Score":
                _z  = z_thresh if method == m else 3.0
                _zw = z_window if method == m else 30
                labels, scores = detect_zscore(values, _z, _zw)
                n_train, offset = None, 0

            elif m == "IQR":
                _f = iqr_factor if method == m else 1.75
                labels, scores = detect_iqr(values, _f)
                n_train, offset = None, 0

            elif m == "Isolation Forest":
                _c = contam if method in (m, "All Methods (ensemble)") else 0.05
                labels, scores = detect_isoforest(values, _c)
                n_train, offset = None, 0

            elif m == "Local Outlier Factor":
                _c = contam if method in (m, "All Methods (ensemble)") else 0.05
                labels, scores = detect_lof(values, _c)
                n_train, offset = None, 0

            else:
                labels, scores = np.zeros(n, dtype=int), np.zeros(n)
                n_train, offset = None, 0

            st.session_state.results[m] = (labels, scores, n_train, offset)
            status.empty()

        # Ensemble: majority vote
        if method == "All Methods (ensemble)":
            votes = np.stack([
                st.session_state.results[m][0]
                for m in methods_to_run
            ]).sum(axis=0)
            ens_labels = (votes >= 3).astype(int)
            ens_scores = np.stack([
                st.session_state.results[m][1]
                for m in methods_to_run
            ]).mean(axis=0)
            st.session_state.results["Ensemble"] = (ens_labels, ens_scores, None, 0)

    st.session_state.last_method = method
    st.rerun()

# ── TAB 2: DETECTION RESULTS ─────────────────────────────────────────────────
with tab_results:
    if not st.session_state.results:
        st.markdown('<div class="info-box">Hit "Run Detection" in the sidebar to see results here.</div>', unsafe_allow_html=True)
    else:
        primary_key = "Ensemble" if "Ensemble" in st.session_state.results else list(st.session_state.results.keys())[0]
        labels, scores, n_train, offset = st.session_state.results[primary_key]

        n_anom = int(labels.sum())
        anom_pct = 100 * n_anom / n

        # Metric cards
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card red"><div class="val">{n_anom}</div><div class="lbl">Anomalies detected</div></div>
          <div class="metric-card blue"><div class="val">{anom_pct:.1f}%</div><div class="lbl">Anomaly rate</div></div>
          <div class="metric-card grn"><div class="val">{n - n_anom:,}</div><div class="lbl">Normal points</div></div>
          <div class="metric-card"><div class="val">{primary_key.split()[0]}</div><div class="lbl">Primary method</div></div>
        </div>
        """, unsafe_allow_html=True)

        fig = plot_results(df, labels, scores, primary_key,
                           n_train=n_train, offset=offset)
        st.plotly_chart(fig, use_container_width=True)

        # Score distribution + training loss side by side
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                plot_score_distribution(scores[scores > 0]),
                use_container_width=True
            )
        with col_r:
            if st.session_state.gru_history:
                st.plotly_chart(
                    plot_training_loss(st.session_state.gru_history),
                    use_container_width=True
                )
            else:
                st.markdown('<div class="info-box" style="margin-top:1rem">Training loss chart only available for GRU Autoencoder.</div>', unsafe_allow_html=True)

        # Anomaly table
        st.markdown("**Detected anomaly timestamps**")
        anom_df = df[labels == 1][["timestamp", "value"]].copy()
        anom_df["anomaly_score"] = scores[labels == 1].round(4)
        anom_df = anom_df.sort_values("anomaly_score", ascending=False)
        st.dataframe(anom_df.reset_index(drop=True), use_container_width=True, height=280)


# ── TAB 3: COMPARISON ─────────────────────────────────────────────────────────
with tab_compare:
    if len(st.session_state.results) < 2:
        st.markdown('<div class="info-box">Select "All Methods (ensemble)" and run detection to see method comparisons.</div>', unsafe_allow_html=True)
    else:
        # Summary table
        rows = []
        for mname, (lbl, sc, _, _) in st.session_state.results.items():
            rows.append({
                "Method": mname,
                "Anomalies": int(lbl.sum()),
                "Rate (%)": round(100 * lbl.sum() / n, 2),
                "Max score": round(sc.max(), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Comparison chart
        compare_results = {
            m: (v[0], v[1])
            for m, v in st.session_state.results.items()
        }
        st.plotly_chart(
            plot_comparison(df, compare_results),
            use_container_width=True
        )


# ── TAB 4: EXPORT ─────────────────────────────────────────────────────────────
with tab_export:
    if not st.session_state.results:
        st.markdown('<div class="info-box">Run detection first to export results.</div>', unsafe_allow_html=True)
    else:
        st.markdown("**Download results CSV**")

        export_df = df[["timestamp", "value"]].copy()
        for mname, (lbl, sc, _, _) in st.session_state.results.items():
            safe = mname.lower().replace(" ", "_")
            export_df[f"{safe}_label"] = lbl
            export_df[f"{safe}_score"] = sc.round(5)

        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download results.csv",
            data=csv_bytes,
            file_name="nab_anomaly_results.csv",
            mime="text/csv",
        )

        st.markdown("**Download detected anomalies only**")
        primary_key = "Ensemble" if "Ensemble" in st.session_state.results else list(st.session_state.results.keys())[0]
        lbl, sc, _, _ = st.session_state.results[primary_key]
        anom_only = df[lbl == 1][["timestamp", "value"]].copy()
        anom_only["score"] = sc[lbl == 1].round(5)

        st.download_button(
            label="⬇️  Download anomalies_only.csv",
            data=anom_only.to_csv(index=False).encode(),
            file_name="anomalies_only.csv",
            mime="text/csv",
        )

        st.markdown("**Preview export**")
        st.dataframe(export_df.head(50), use_container_width=True)
