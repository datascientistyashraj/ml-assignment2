"""
app.py - Streamlit front end for the five saved classifiers.

The app never trains anything. It loads the joblib files in model/, scores the test CSV you upload
(or the bundled test_data.csv) and shows the results. Run it with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

import ml_core as core

# ---- design tokens ------------------------------------------------------------------------------
INK, MUTED, LINE = "#1F2A44", "#5C6B80", "#D5DBE3"
HIT, MISS = "#0F766E", "#C77D0A"          # teal = correct prediction, amber = mistake
SAMPLE_PATH = Path(__file__).resolve().parent / "test_data.csv"

st.set_page_config(page_title="Classifier comparison", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
    html, body, .stApp, [data-testid="stSidebar"], .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp button, .stApp input, .stTabs [data-baseweb="tab"] {
        font-family: 'Atkinson Hyperlegible', 'Segoe UI', system-ui, sans-serif;
    }
    .stApp h1 { font-weight: 700; font-size: 2.05rem; line-height: 1.2; letter-spacing: -0.01em; }
    .stApp h2, .stApp h3 { font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    .metric-strip { display: flex; flex-wrap: wrap; gap: 0.6rem 2.4rem; margin: 0.2rem 0 0.8rem 0; }
    .metric-label { font-size: 0.95rem; opacity: 0.7; }
    .metric-value { font-size: 2.1rem; font-weight: 700; line-height: 1.15; white-space: nowrap;
                    font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- cached loading and scoring ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Saved metadata plus the five fitted pipelines (loaded once per server process)."""
    return core.load_metadata(), core.load_models()


@st.cache_data(show_spinner="Scoring the five models...")
def score_all(df: pd.DataFrame):
    """Integer labels and the full evaluation of every saved model on `df` (no fitting)."""
    meta, models = get_artifacts()
    X, y = core.split_xy(df, meta)
    return y, core.evaluate_all(models, X, y, meta)


# ---- small helpers -----------------------------------------------------------------------------------
def read_csv_safely(source):
    try:
        return pd.read_csv(source), None
    except Exception as exc:  # unreadable bytes, empty file, broken quoting ...
        return None, f"The file could not be read as a CSV ({exc.__class__.__name__}: {exc})."


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def metric_strip_html(metrics: dict) -> str:
    """The six scores as one row of big numbers. Unlike st.metric, nothing is cut off with '...'; the row wraps instead."""
    cells = "".join(
        f'<div class="metric-cell"><div class="metric-label">{name}</div>'
        f'<div class="metric-value">{"n/a" if pd.isna(value) else f"{value:.4f}"}</div></div>'
        for name, value in metrics.items())
    return f'<div class="metric-strip">{cells}</div>'


def prediction_table(y, res, meta) -> pd.DataFrame:
    classes = np.array(meta["classes"])
    n = len(y)
    return pd.DataFrame({
        "row": np.arange(1, n + 1),
        "actual": classes[y],
        "predicted": classes[res["y_pred"]],
        "confidence": res["proba"][np.arange(n), res["y_pred"]].round(4),
        "correct": y == res["y_pred"],
    })


def top_confusions(cm, classes, k=6) -> pd.DataFrame:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(classes[i], classes[j], int(off[i, j]))
             for i in range(len(classes)) for j in range(len(classes)) if off[i, j] > 0]
    pairs.sort(key=lambda t: -t[2])
    return pd.DataFrame(pairs[:k], columns=["Actual class", "Predicted as", "Rows"])


def confusion_figure(cm, classes, as_percent):
    """Teal cells (diagonal) are correct predictions; amber cells are mistakes. Darker = more rows."""
    n = len(classes)
    values = cm.astype(float)
    if as_percent:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    hit = LinearSegmentedColormap.from_list("hit", ["#E3F1EF", HIT])
    miss = LinearSegmentedColormap.from_list("miss", ["#FBEBCB", MISS])
    top_hit = max(float(np.diag(values).max()), 1e-9)
    off = values - np.diag(np.diag(values))
    top_miss = max(float(off.max()), 1e-9)
    rgb = np.ones((n, n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                rgb[i, j] = hit(values[i, j] / top_hit)[:3]
            elif values[i, j] > 0:
                rgb[i, j] = miss(values[i, j] / top_miss)[:3]
            else:
                rgb[i, j] = (0.98, 0.985, 0.99)
    side = min(8.0, max(4.6, 0.62 * n + 2.2))
    fig, ax = plt.subplots(figsize=(side, side * 0.92))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45 if max(len(c) for c in classes) > 3 else 0, ha="right" if max(len(c) for c in classes) > 3 else "center", color=INK)
    ax.set_yticklabels(classes, color=INK)
    ax.set_xlabel("Predicted class", color=INK)
    ax.set_ylabel("Actual class", color=INK)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                v = values[i, j]
                if i != j and v == 0:
                    continue
                dark_fill = (i == j and v / top_hit > 0.55) or (i != j and v / top_miss > 0.75)
                text = f"{v:.0%}" if as_percent else f"{int(v)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9 if n <= 10 else 7.5,
                        color="white" if dark_fill else INK)
    fig.tight_layout()
    return fig


def metric_bars(series: pd.Series, metric: str):
    s = series.dropna().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * max(len(s), 1) + 1.0))
    best = s.max()
    bars = ax.barh(s.index, s.values, color=[HIT if v == best else "#9AA8B8" for v in s.values], height=0.58)
    low = min(0.0, float(s.min()) - 0.05)
    ax.set_xlim(low, 1.08)
    for bar, v in zip(bars, s.values):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlabel(metric, color=INK)
    ax.tick_params(colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    fig.tight_layout()
    return fig


# ---- load the saved models -------------------------------------------------------------------------
try:
    meta, models = get_artifacts()
except Exception as exc:
    st.title("Classifier comparison")
    st.error("The saved models could not be loaded.")
    st.write(f"{exc.__class__.__name__}: {exc}")
    st.write("Check that the model/ folder is in the repository. If you changed Python or scikit-learn "
             "versions, run `python train_models.py` again in the same environment and commit the new files.")
    st.stop()

classes = meta["classes"]
st.title(f"{meta['dataset_name']}: how five classifiers compare")
st.caption("Upload a test CSV, choose a model, and see where it gets things right and where it confuses classes. "
           "The models were trained beforehand; this page only scores them.")

# ---- sidebar -------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV", type=["csv"],
        help=f"The file needs the {len(meta['features'])} feature columns and the '{meta['target']}' column, "
             "with the same names as the bundled test set.")
    st.caption("Results appear as soon as a file is uploaded. There is no run button.")
    use_sample = st.checkbox("Use the bundled test set when no file is uploaded", value=True)
    if SAMPLE_PATH.exists():
        st.download_button("Download the bundled test set", SAMPLE_PATH.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    st.header("Model")
    model_name = st.selectbox("Classifier", list(core.MODEL_FILES))
    st.header("Display")
    as_percent = st.toggle("Show the confusion matrix as row percentages", value=False)

# ---- choose and validate the data ------------------------------------------------------------------
if uploaded is not None:
    df, read_error = read_csv_safely(uploaded)
    source_label = f"your file, {uploaded.name}"
elif use_sample and SAMPLE_PATH.exists():
    df, read_error = read_csv_safely(SAMPLE_PATH)
    source_label = "the bundled test set"
else:
    st.info("No test data yet. Upload a CSV in the sidebar, or turn on the bundled test set to see the models at work.")
    st.stop()

if read_error:
    st.error(read_error)
    st.write("Save the file as a comma-separated CSV with a header row and try again.")
    st.stop()

problems = core.check_test_frame(df, meta)
if problems:
    st.error("This file cannot be scored yet.")
    for problem in problems:
        st.write(f"- {problem}")
    st.write("Use the same column names as the bundled test set. Download it from the sidebar to compare.")
    with st.expander("Columns the models expect"):
        st.write(", ".join(meta["features"] + [meta["target"]]))
    st.stop()

try:
    y, results = score_all(df)
except Exception as exc:
    st.error("The models could not score this file.")
    st.write(f"{exc.__class__.__name__}: {exc}")
    st.write("Check that numeric columns contain only numbers and that text columns hold the same kinds of "
             "values as the bundled test set.")
    st.stop()

extra = [c for c in df.columns if c not in meta["features"] and c != meta["target"]]
st.write(f"Scoring {len(df):,} rows from {source_label}."
         + (f" {len(extra)} extra column(s) in the file are ignored." if extra else ""))

tab_report, tab_compare, tab_data, tab_about = st.tabs(["Model report", "Compare models", "Your data", "About"])

# ---- tab 1: one model in detail -----------------------------------------------------------------------
with tab_report:
    res = results[model_name]
    st.subheader(model_name)
    st.markdown(metric_strip_html({key: res["metrics"][key] for key in core.METRICS}), unsafe_allow_html=True)
    present = int(len(np.unique(y)))
    if present < 2:
        st.caption("The file holds a single class, so AUC is n/a and MCC is reported as 0 (both need at least two classes).")
    elif pd.isna(res["metrics"]["AUC"]):
        st.caption("AUC is n/a for this data.")
    elif present < len(classes):
        st.caption(f"The uploaded rows cover {present} of the {len(classes)} classes, so class-averaged scores and "
                   "AUC use only the classes that appear.")
    if meta["task"] == "binary":
        st.caption(f"Precision, recall and F1 describe the positive class, '{meta['positive_class']}'.")
    else:
        st.caption("Precision, recall and F1 are averages over classes, weighted by how many rows each class has. "
                   "AUC compares each class against the rest.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Confusion matrix**")
        fig = confusion_figure(res["confusion"], classes, as_percent)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Teal cells are correct predictions. Amber cells are mistakes, darker where more rows are confused.")
    with right:
        st.markdown("**Classification report**")
        st.dataframe(res["report"].round(3))
        st.markdown("**Most frequent mistakes**")
        confusions = top_confusions(res["confusion"], classes)
        if confusions.empty:
            st.write("No mistakes on this data.")
        else:
            st.dataframe(confusions, hide_index=True)

    table = prediction_table(y, res, meta)
    wrong = table[~table["correct"]]
    with st.expander(f"Misclassified rows ({len(wrong)})"):
        if wrong.empty:
            st.write("Every row was classified correctly.")
        else:
            st.dataframe(wrong, hide_index=True)
    st.download_button("Download all predictions (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{slug(model_name)}.csv", mime="text/csv")

# ---- tab 2: all five models side by side -----------------------------------------------------------------
with tab_compare:
    board = core.leaderboard(results)
    rank_by = st.selectbox("Rank models by", core.METRICS, index=core.METRICS.index("MCC"))
    ordered = board.sort_values(rank_by, ascending=False, na_position="last")
    st.dataframe(ordered.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#CDE7E3"))
    st.caption("Highlighted cells hold the best value in each column.")
    ranks = core.mean_ranks(board)
    st.markdown(f"**Best overall on this data:** {ranks.index[0]}, with an average rank of {ranks.iloc[0]:.2f} "
                "across the six metrics (1 is best).")
    fig = metric_bars(board[rank_by], rank_by)
    st.pyplot(fig)
    plt.close(fig)

# ---- tab 3: the uploaded data ------------------------------------------------------------------------------
with tab_data:
    st.write(f"{len(df):,} rows and {df.shape[1]} columns.")
    counts = df[meta["target"]].map(core.label_to_str).value_counts().reindex(classes).fillna(0).astype(int)
    st.markdown("**Rows per class**")
    st.bar_chart(counts)
    missing = df[meta["features"]].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.write("No missing values in the feature columns.")
    else:
        st.write("Missing values are filled with the training median (numbers) or most frequent value (text).")
        st.dataframe(missing.rename("Missing values"))
    st.markdown("**First 25 rows**")
    st.dataframe(df.head(25), hide_index=True)

# ---- tab 4: how it was built --------------------------------------------------------------------------------
with tab_about:
    st.markdown(f"**Data.** {meta['dataset_name']}. Source: {meta['source']}. "
                f"{meta['n_rows']:,} rows, {len(meta['features'])} features and {meta['n_classes']} classes.")
    st.markdown(f"**Split.** A stratified {int(round((1 - meta['test_size']) * 100))}/{int(round(meta['test_size'] * 100))} "
                f"split (random_state {meta['random_state']}): {meta['n_train']:,} rows for training and "
                f"{meta['n_test']:,} for testing. Hyper-parameters were tuned with 5-fold cross-validation on the "
                "training rows only, so the test rows were used once, at the end.")
    st.markdown("**Chosen hyper-parameters**")
    params = pd.DataFrame({"Model": list(meta["best_params"]),
                           "Settings": [", ".join(f"{k} = {v}" for k, v in p.items()) or "scikit-learn defaults"
                                        for p in meta["best_params"].values()]})
    st.dataframe(params, hide_index=True)
    st.markdown(
        "**What the metrics mean**\n\n"
        "- Accuracy: the share of rows predicted correctly.\n"
        "- AUC: how reliably the predicted probabilities rank the true class above the others.\n"
        "- Precision: of the rows predicted as a class, the share that truly belong to it.\n"
        "- Recall: of the rows that truly belong to a class, the share the model found.\n"
        "- F1: the harmonic mean of precision and recall.\n"
        "- MCC: the correlation between predictions and truth, from -1 to 1. It stays honest when classes are unbalanced."
    )
    st.caption(f"Models were trained with scikit-learn {meta.get('sklearn_version', 'unknown')}; "
               f"this page is running scikit-learn {sklearn.__version__}.")
