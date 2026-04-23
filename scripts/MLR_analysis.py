import evofr as ef
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.nn import softmax
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
from evofr import VariantFrequencies, MultinomialLogisticRegression, InferFullRank
from evofr.plotting import plot_posterior_frequency, plot_observed_frequency, plot_growth_advantage, add_dates_sep
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

from matplotlib.backends.backend_pdf import PdfPages

# Read full dataset (state-level)
raw_seq = pd.read_csv("/scratch/gpfs/ez1199/usa.tsv", sep="\t")
raw_seq["date"] = pd.to_datetime(raw_seq["date"])
raw_seq.rename(columns={"clade": "variant"}, inplace=True)
raw_seq = raw_seq.groupby(["date", "variant"], as_index=False)["sequences"].sum()

# Set font
font = {'family': 'sans-serif', 'weight': 'light', 'size': 10}
matplotlib.rc('font', **font)

# Create color map
all_variants = raw_seq["variant"].unique()
color_palette = plt.cm.tab20.colors
color_map = {v: matplotlib.colors.to_hex(color_palette[i % len(color_palette)]) for i, v in enumerate(all_variants)}

def forecast_frequencies(samples, mlr, forecast_L):
    last_T = samples["freq"].shape[1]
    X = mlr.make_ols_feature(start=last_T, stop=last_T + forecast_L)
    beta = jnp.array(samples["beta"])
    logits = vmap(jnp.dot, in_axes=(None, 0))(X, beta)
    return softmax(logits, axis=-1)

def generate_rolling_windows(df, window_size_days=60, step_size_days=60, min_variant_sequences=250):
    global_variant_counts = df.groupby("variant")["sequences"].sum()
    globally_common_variants = global_variant_counts[global_variant_counts > 1000].index.tolist()
    df = df[df["variant"].isin(globally_common_variants)]

    min_date = df.date.min()
    max_date = df.date.max()
    windows = []
    start = min_date

    while start + pd.Timedelta(days=window_size_days) <= max_date:
        end = start + pd.Timedelta(days=window_size_days)
        window_df = df[(df.date >= start) & (df.date < end)].copy()

        variant_totals = window_df.groupby("variant")["sequences"].sum()
        passing_variants = variant_totals[variant_totals >= min_variant_sequences].index.tolist()
        window_df = window_df[window_df["variant"].isin(passing_variants)]

        if len(window_df) > 0:
            windows.append((start, end, window_df))

        start += pd.Timedelta(days=step_size_days)

    return windows

def plot_windowed_frequency(window_df, start, end, color_map, ax):
    vf_window = VariantFrequencies(window_df)
    mlr = MultinomialLogisticRegression(tau=4.2)
    inference = InferFullRank(iters=10000, lr=0.01, num_samples=100)
    posterior = inference.fit(mlr, vf_window)
    samples = posterior.samples

    forecast_L = 30
    samples["freq_forecast"] = forecast_frequencies(samples, mlr, forecast_L)

    ps = [0.95, 0.8, 0.5]
    alphas = [0.2, 0.4, 0.6]
    colors = [color_map.get(v, "#999999") for v in vf_window.var_names]

    plot_posterior_frequency(ax, samples, ps, alphas, colors, forecast=True)
    plot_observed_frequency(ax, vf_window, colors)

    ax.axvline(x=len(vf_window.dates) - 1, color='k', linestyle='--')
    add_dates_sep(ax, ef.data.expand_dates(vf_window.dates, forecast_L), sep=20)
    ax.set_ylabel("Variant frequency")
    ax.set_title(f"{start.date()} to {end.date()}")

def plot_windowed_growth(window_df, start, end, color_map, ax):
    variant_counts = window_df.groupby("variant")["sequences"].sum()
    ref_variant = variant_counts.idxmax()
    ref_count = variant_counts[ref_variant]

    vf_window = VariantFrequencies(window_df, pivot=ref_variant)
    mlr = MultinomialLogisticRegression(tau=4.2)
    inference = InferFullRank(iters=10000, lr=0.01, num_samples=100)
    posterior = inference.fit(mlr, vf_window)
    samples = posterior.samples

    counts = vf_window.seq_counts
    frequencies = counts / counts.sum(axis=1, keepdims=True)
    avg_freqs = frequencies.mean(axis=0)
    freq_dict = dict(zip(vf_window.var_names, avg_freqs))

    ga = samples["ga"]
    means = np.mean(ga, axis=0)
    lower = np.percentile(ga, 2.5, axis=0)
    upper = np.percentile(ga, 97.5, axis=0)

    summary_lines = [f"Growth Advantage Summary for window {start.date()} to {end.date()}:"]
    summary_lines.append(f"Reference variant: {ref_variant}")

    for name, mean, lo, hi in zip(vf_window.var_names[:-1], means, lower, upper):
        summary_lines.append(f"{name:>5} | mean: {mean:.3f}, 95% CI: [{lo:.3f}, {hi:.3f}]")

    handles = [matplotlib.lines.Line2D([0], [0], color=color_map.get(v, "#999999"), lw=2, label=v)
               for v in vf_window.var_names]
    ax.legend(handles=handles, title="Variants", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    colors = [color_map.get(v, "#999999") for v in vf_window.var_names]
    plot_growth_advantage(ax, samples, vf_window, [0.95, 0.8, 0.5], [0.2, 0.4, 0.6], colors)

    ax.set_ylabel("Growth advantage")
    ax.set_title(f"Growth Advantage: {start.date()} to {end.date()}")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    return "\n".join(summary_lines)

windows = generate_rolling_windows(raw_seq)

all_plotted_variants = set()
summary_texts = []  # Collect all summaries here

num_windows = len(windows)
fig = plt.figure(figsize=(20, 8 * num_windows))
gs = gridspec.GridSpec(
    nrows=2 * num_windows, 
    ncols=2, 
    height_ratios=[0.3, 1.5] * num_windows
)

ax_header = fig.add_subplot(gs[0, :])
ax_header.axis("off")

for i, (start, end, window_df) in enumerate(windows):
    ax_freq = fig.add_subplot(gs[2 * i + 1, 0])
    ax_growth = fig.add_subplot(gs[2 * i + 1, 1])

    if window_df["variant"].nunique() < 2:
        ax_freq.axis("off")
        ax_growth.axis("off")
        continue

    all_plotted_variants.update(window_df["variant"].unique())

    plot_windowed_frequency(window_df, start, end, color_map, ax_freq)
    summary_text = plot_windowed_growth(window_df, start, end, color_map, ax_growth)

    summary_texts.append(summary_text)

ax_header.text(
    0.5, 0.5,
    f"Total unique variants plotted across all windows: {len(all_plotted_variants)}",
    ha='center',
    va='center',
    fontsize=12,
    weight='bold'
)

plt.tight_layout()
plt.savefig("growth_advantage_2mo_usa_plots.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Save summaries in a separate PDF

with PdfPages("growth_advantage_2mo_usa_summaries.pdf") as pdf:
    for summary in summary_texts:
        fig_summary, ax_summary = plt.subplots(figsize=(8.5, 11))
        ax_summary.axis("off")
        ax_summary.text(0.5, 0.5, summary, ha='center', va='center', fontsize=10, family='monospace', wrap=True)
        pdf.savefig(fig_summary)
        plt.close(fig_summary)
