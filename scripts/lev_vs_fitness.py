import torch
from Bio import SeqIO
import re
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import linregress
import jax.numpy as jnp
from jax import vmap
import evofr as ef
from jax.nn import softmax
import matplotlib
from matplotlib import gridspec
from evofr import VariantFrequencies, MultinomialLogisticRegression, InferFullRank
from evofr.plotting import plot_posterior_frequency, plot_observed_frequency, plot_growth_advantage, add_dates_sep
import warnings
warnings.filterwarnings("ignore")
import gc
from Levenshtein import distance as levenshtein_distance  # You'll need to install python-Levenshtein

# Global setup
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
font = {'family': 'sans-serif', 'weight': 'light', 'size': 10}
matplotlib.rc('font', **font)

def analyze_lineage_fitness(parent_lineage, lineages, start_date, end_date, protein="surface glycoprotein"):
    """
    Analyze the relationship between protein Levenshtein distances and fitness differences
    for given lineages within a specified time window.
    
    Args:
        parent_lineage (str): The reference lineage (e.g., "B.1")
        lineages (list): List of lineages to analyze (including parent)
        start_date (str): Start date in "YYYY-MM-DD" format
        end_date (str): End date in "YYYY-MM-DD" format
        protein (str): Protein name for file paths
    
    Returns:
        pd.DataFrame: Results containing Levenshtein distances and fitness differences
    """
    
    print(f"Processing {parent_lineage} from {start_date} to {end_date}")
    
    try:
        # === PROTEIN SEQUENCE SECTION (REPLACED EMBEDDING LOGIC) ===
        sequences = []
        sequence_ids = []
        files = [
            f"/scratch/gpfs/GRENFELL/ez1199/test_data/{parent_lineage}_{protein.replace(' ', '_')}_{start_date}to{end_date}/{lineage}.fa"
            for lineage in lineages
        ]

        # Load sequences
        for file in files:
            if not os.path.exists(file):
                print(f"Warning: {file} not found. Skipping.")
                continue

            lineage = os.path.splitext(os.path.basename(file))[0]
            for record in SeqIO.parse(file, "fasta"):
                # Clean sequence but keep as string (no spacing needed for Levenshtein)
                cleaned_seq = re.sub(r"[UZOB]", "X", str(record.seq))
                sequences.append(cleaned_seq)
                sequence_ids.append(lineage)

        if not sequences:
            print("No sequences found. Skipping this parameter set.")
            return pd.DataFrame()

        # Group sequences by lineage
        lineage_to_sequences = defaultdict(list)
        for seq, lin in zip(sequences, sequence_ids):
            lineage_to_sequences[lin].append(seq)

        # Clear original sequences to free memory
        del sequences, sequence_ids
        gc.collect()

        # Calculate Levenshtein distances
        if parent_lineage not in lineage_to_sequences:
            print(f"Parent lineage {parent_lineage} not found in sequences. Skipping.")
            return pd.DataFrame()

        # Get reference sequences for parent lineage
        ref_sequences = lineage_to_sequences[parent_lineage]
        
        ref_consensus = ref_sequences[0]

        child_lineages = []
        levenshtein_distances = []

        for lineage, seqs in lineage_to_sequences.items():
            if lineage == parent_lineage:
                continue
            if seqs[0] == ref_consensus:
                print(f"Skipping {lineage}: identical sequence to parent {parent_lineage}")
                continue

            child_lineages.append(lineage)
            levenshtein_distances.append(levenshtein_distance(ref_consensus, seqs[0]))

        # Clear sequence data
        del lineage_to_sequences, ref_sequences
        gc.collect()
        
        # === FITNESS ANALYSIS SECTION (UNCHANGED) ===
        raw_seq = pd.read_csv("/scratch/gpfs/GRENFELL/ez1199/usa.tsv", sep="\t")
        raw_seq["date"] = pd.to_datetime(raw_seq["date"])
        raw_seq.rename(columns={"clade": "variant"}, inplace=True)
        raw_seq = raw_seq.groupby(["date", "variant"], as_index=False)["sequences"].sum()

        # Create color map
        all_variants = raw_seq["variant"].unique()
        color_palette = plt.cm.tab20.colors
        color_map = {v: matplotlib.colors.to_hex(color_palette[i % len(color_palette)]) 
                    for i, v in enumerate(all_variants)}

        def forecast_frequencies(samples, mlr, forecast_L):
            last_T = samples["freq"].shape[1]
            X = mlr.make_ols_feature(start=last_T, stop=last_T + forecast_L)
            beta = jnp.array(samples["beta"])
            logits = vmap(jnp.dot, in_axes=(None, 0))(X, beta)
            return softmax(logits, axis=-1)

        def plot_windowed_growth(window_df, start, end, color_map):
            variant_counts = window_df.groupby("variant")["sequences"].sum()
            ref_variant = variant_counts.idxmax()

            vf_window = VariantFrequencies(window_df, pivot=ref_variant)
            mlr = MultinomialLogisticRegression(tau=4.2)
            inference = InferFullRank(iters=10000, lr=0.01, num_samples=100)
            posterior = inference.fit(mlr, vf_window)
            samples = posterior.samples

            ga = samples["ga"]
            means = np.mean(ga, axis=0)

            growth_advantage_dict = {name: mean for name, mean in zip(vf_window.var_names[:-1], means)}
            growth_advantage_dict[ref_variant] = 1.0

            ga_samples_dict = {name: ga[:, i] for i, name in enumerate(vf_window.var_names[:-1])}
            ga_samples_dict[ref_variant] = np.ones(ga.shape[0])

            return growth_advantage_dict, ga_samples_dict

        # Process time window
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        global_variant_counts = raw_seq.groupby("variant")["sequences"].sum()
        globally_common_variants = global_variant_counts[global_variant_counts > 1000].index.tolist()
        filtered_df = raw_seq[raw_seq["variant"].isin(globally_common_variants)]

        window_df = filtered_df[(filtered_df.date >= start) & (filtered_df.date < end)].copy()
        variant_totals = window_df.groupby("variant")["sequences"].sum()
        passing_variants = variant_totals[variant_totals >= 250].index.tolist()
        window_df = window_df[window_df["variant"].isin(passing_variants)]

        # Clear large dataframes
        del raw_seq, filtered_df
        gc.collect()

        if window_df["variant"].nunique() < 2:
            print(f"Fewer than 2 variants for {parent_lineage} {start_date}-{end_date}. Skipping.")
            return pd.DataFrame()

        growth_advantage_dict, ga_samples_dict = plot_windowed_growth(window_df, start, end, color_map)

        # === COMBINE RESULTS ===
        # Only keep children that appear in the growth advantage model
        paired = [(lin, dist) for lin, dist in zip(child_lineages, levenshtein_distances)
                  if lin in growth_advantage_dict]
        if not paired:
            print(f"No child lineages found in growth advantage model for {parent_lineage} {start_date}-{end_date}. Skipping.")
            return pd.DataFrame()
        child_lineages, levenshtein_distances = zip(*paired)

        result_df = pd.DataFrame({
            "Parent Lineage": [parent_lineage] * len(child_lineages),
            "Child Lineage": list(child_lineages),
            "Levenshtein Distance": list(levenshtein_distances),
        })

        fitness_differences = []
        fitness_lower = []
        fitness_upper = []
        for child in result_df["Child Lineage"]:
            child_ga = growth_advantage_dict.get(child, np.nan)
            parent_ga = growth_advantage_dict.get(parent_lineage, np.nan)

            if parent_ga == 1.0:
                raw_ratio = child_ga
            else:
                raw_ratio = child_ga / parent_ga if (not np.isnan(parent_ga) and not np.isnan(child_ga) and parent_ga != 0) else np.nan

            if not np.isnan(raw_ratio) and raw_ratio > 0:
                fitness_differences.append(np.log(raw_ratio))
            else:
                fitness_differences.append(np.nan)

            child_samps = ga_samples_dict.get(child)
            parent_samps = ga_samples_dict.get(parent_lineage)
            if child_samps is not None and parent_samps is not None:
                ratio_samps = child_samps if growth_advantage_dict.get(parent_lineage) == 1.0 else child_samps / parent_samps
                log_samps = np.log(ratio_samps[ratio_samps > 0])
                if len(log_samps) > 0:
                    fitness_lower.append(float(np.percentile(log_samps, 2.5)))
                    fitness_upper.append(float(np.percentile(log_samps, 97.5)))
                else:
                    fitness_lower.append(np.nan)
                    fitness_upper.append(np.nan)
            else:
                fitness_lower.append(np.nan)
                fitness_upper.append(np.nan)

        result_df["Fitness Difference"] = fitness_differences
        result_df["Fitness Lower"] = fitness_lower
        result_df["Fitness Upper"] = fitness_upper

        valid_x = result_df.dropna(subset=["Levenshtein Distance", "Fitness Difference"])["Levenshtein Distance"].astype(float).to_numpy()
        if len(np.unique(valid_x)) <= 1:
            print(f"Insufficient x variation for regression: {parent_lineage} {start_date}-{end_date}. Skipping.")
            return pd.DataFrame()

        # Save results
        output_dir = "levsummary_test"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"summary_{parent_lineage}_{start_date}to{end_date}_{protein.replace(' ', '_')}_levenshtein.csv"  # Added suffix
        result_df.to_csv(os.path.join(output_dir, filename), index=False)

        # Create plot
        create_fitness_plot(result_df, parent_lineage, start_date, end_date, protein)
        
        print(f"Completed {parent_lineage} from {start_date} to {end_date}")
        return result_df

    except Exception as e:
        print(f"Error in analyze_lineage_fitness: {e}")
        return pd.DataFrame()


def create_fitness_plot(result_df, parent_lineage, start_date, end_date, protein):
    """Create and save the fitness vs Levenshtein distance plot"""
    
    # Clean data - Updated column name
    result_df["Levenshtein Distance"] = pd.to_numeric(result_df["Levenshtein Distance"], errors='coerce')
    result_df["Fitness Difference"] = pd.to_numeric(result_df["Fitness Difference"], errors='coerce')
    for col in ["Fitness Lower", "Fitness Upper"]:
        if col not in result_df.columns:
            result_df[col] = np.nan
    clean_df = result_df.dropna(subset=["Levenshtein Distance", "Fitness Difference", "Child Lineage", "Parent Lineage"])

    if len(clean_df) == 0:
        print("No valid data for plotting")
        return

    x = clean_df["Levenshtein Distance"].astype(float).to_numpy()
    y = clean_df["Fitness Difference"].astype(float).to_numpy()

    if len(np.unique(x)) <= 1:
        print(f"Insufficient x values for plotting: {parent_lineage} {start_date}-{end_date}. Skipping.")
        return

    plt.figure(figsize=(10, 6))

    unique_lineages = clean_df["Child Lineage"].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_lineages))

    for i, lineage in enumerate(unique_lineages):
        subset = clean_df[clean_df["Child Lineage"] == lineage]
        y_mid = subset["Fitness Difference"].astype(float).to_numpy()
        y_lo = subset["Fitness Lower"].astype(float).to_numpy()
        y_hi = subset["Fitness Upper"].astype(float).to_numpy()
        yerr_lo = np.where(np.isnan(y_lo), 0, y_mid - y_lo)
        yerr_hi = np.where(np.isnan(y_hi), 0, y_hi - y_mid)
        plt.errorbar(
            subset["Levenshtein Distance"],
            y_mid,
            yerr=[yerr_lo, yerr_hi],
            fmt='o',
            color=colors(i),
            label=lineage,
            ecolor=colors(i),
            elinewidth=1,
            capsize=3,
            markersize=7,
            markeredgecolor='black',
            markeredgewidth=0.5,
            alpha=0.8
        )

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2

    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = slope * x_vals + intercept
    r2_line, = plt.plot(x_vals, y_vals, color='black', linestyle='--')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(r2_line)
    labels.append(f"Fit: $R^2$ = {r_squared:.2f}")
    plt.legend(handles, labels, title="Child Lineage", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Levenshtein Distance")  # Updated label
    plt.ylabel("Log Relative Fitness Change")
    plt.title(f"Log Relative Fitness Change vs Levenshtein Distance\n{start_date} to {end_date}, {parent_lineage} Parent Lineage, {protein}", fontsize=10)
    plt.tight_layout()

    output_dir = "levplots_test"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"plot_{parent_lineage}_{start_date}to{end_date}_{protein.replace(' ', '_')}_levenshtein.png"  # Added suffix
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define all 44 parameter sets
    parameter_sets = [
        {
            "parent_lineage": "B.1",
            "lineages": ["B.1", "B.1.1", "B.1.2", "B.1.206", "B.1.240", "B.1.243", "B.1.371", "B.1.426"],
            "start_date": "2020-05-03",
            "end_date": "2020-07-02",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1",
            "lineages": ["B.1", "B.1.1", "B.1.2", "B.1.206", "B.1.240", "B.1.243", "B.1.400", "B.1.426", "B.1.509", "B.1.565"],
            "start_date": "2020-07-02",
            "end_date": "2020-08-31",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1",
            "lineages": ["B.1", "B.1.1", "B.1.126", "B.1.234", "B.1.240", "B.1.243", "B.1.396", "B.1.400", "B.1.509", "B.1.565"],
            "start_date": "2020-08-31",
            "end_date": "2020-10-30",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1",
            "lineages": ["B.1", "B.1.1", "B.1.110.3", "B.1.126", "B.1.232", "B.1.234", "B.1.240", "B.1.241", "B.1.243", "B.1.311", "B.1.349", "B.1.396", "B.1.400", "B.1.409", "B.1.427", "B.1.429", "B.1.436", "B.1.509", "B.1.517", "B.1.561", "B.1.565", "B.1.577", "B.1.588", "B.1.609"],
            "start_date": "2020-10-30",
            "end_date": "2020-12-29",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1",
            "lineages": ["B.1", "B.1.1", "B.1.126", "B.1.232", "B.1.234", "B.1.240", "B.1.241", "B.1.243", "B.1.311", "B.1.349", "B.1.396", "B.1.400", "B.1.409", "B.1.427", "B.1.429", "B.1.517", "B.1.526", "B.1.561", "B.1.565", "B.1.575", "B.1.577", "B.1.588", "B.1.609", "B.1.623", "B.1.637"],
            "start_date": "2020-12-29",
            "end_date": "2021-02-27",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1",
            "lineages": ["B.1", "B.1.1", "B.1.2", "B.1.234", "B.1.243", "B.1.311", "B.1.351", "B.1.427", "B.1.429", "B.1.517", "B.1.525", "B.1.526", "B.1.575", "B.1.621", "B.1.623", "B.1.637"],
            "start_date": "2021-02-27",
            "end_date": "2021-04-28",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1.1",
            "lineages": ["B.1.1", "B.1.1.207", "B.1.1.316", "B.1.1.416", "B.1.1.434", "B.1.1.519", "B.1.1.7"],
            "start_date": "2020-12-29",
            "end_date": "2021-02-27",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1.617.2",
            "lineages": ["B.1.617.2", "AY.2", "AY.3", "AY.13", "AY.14", "AY.25", "AY.26", "AY.39", "AY.44", "AY.47", "AY.54", "AY.75", "AY.103", "AY.117", "AY.122"],
            "start_date": "2021-04-28",
            "end_date": "2021-06-27",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1.617.2",
            "lineages": ["B.1.617.2", "AY.1", "AY.2", "AY.3", "AY.4", "AY.5", "AY.13", "AY.14", "AY.20", "AY.25", "AY.26", "AY.33", "AY.35", "AY.37", "AY.39", "AY.43", "AY.44", "AY.47", "AY.48", "AY.52", "AY.54", "AY.62", "AY.64", "AY.67", "AY.75", "AY.98", "AY.100", "AY.103", "AY.107", "AY.110", "AY.113", "AY.114", "AY.117", "AY.118", "AY.119", "AY.122"],
            "start_date": "2021-06-27",
            "end_date": "2021-08-26",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1.617.2",
            "lineages": ["B.1.617.2", "AY.1", "AY.2", "AY.3", "AY.4", "AY.5", "AY.13", "AY.14", "AY.20", "AY.25", "AY.26", "AY.33", "AY.35", "AY.36", "AY.37", "AY.39", "AY.43", "AY.44", "AY.47", "AY.54", "AY.62", "AY.64", "AY.67", "AY.75", "AY.98", "AY.100", "AY.103", "AY.107", "AY.110", "AY.113", "AY.114", "AY.117", "AY.118", "AY.119", "AY.121", "AY.122", "AY.125", "AY.127"],
            "start_date": "2021-08-26",
            "end_date": "2021-10-25",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1.617.2",
            "lineages": ["B.1.617.2", "AY.1", "AY.3", "AY.4", "AY.5", "AY.13", "AY.14", "AY.20", "AY.25", "AY.26", "AY.33", "AY.36", "AY.39", "AY.43", "AY.44", "AY.47", "AY.54", "AY.64", "AY.75", "AY.98", "AY.100", "AY.103", "AY.107", "AY.113", "AY.114", "AY.117", "AY.118", "AY.119", "AY.121", "AY.122", "AY.125", "AY.127"],
            "start_date": "2021-10-25",
            "end_date": "2021-12-24",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "B.1.617.2",
            "lineages": ["B.1.617.2", "AY.100", "AY.103", "AY.117", "AY.119", "AY.25", "AY.3", "AY.39", "AY.44"],
            "start_date": "2021-12-24",
            "end_date": "2022-02-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.1",
            "lineages": ["BA.1", "BA.1.1", "BA.1.15", "BA.1.17", "BA.1.18", "BA.1.20"],
            "start_date": "2021-10-25",
            "end_date": "2021-12-24",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.1",
            "lineages": ["BA.1", "BA.1.15", "BA.1.17", "BA.1.18", "BA.1.20"],
            "start_date": "2021-12-24",
            "end_date": "2022-02-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.1",
            "lineages": ["BA.1", "BA.1.1", "BA.1.15", "BA.1.18", "BA.1.20"],
            "start_date": "2022-02-22",
            "end_date": "2022-04-23",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.2",
            "lineages": ["BA.2", "BA.2.10", "BA.2.3", "BA.2.9"],
            "start_date": "2021-12-24",
            "end_date": "2022-02-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.2",
            "lineages": ["BA.2", "BA.2.1", "BA.2.10", "BA.2.12", "BA.2.18", "BA.2.21", "BA.2.23", "BA.2.26", "BA.2.3", "BA.2.37", "BA.2.65", "BA.2.7", "BA.2.9"],
            "start_date": "2022-02-22",
            "end_date": "2022-04-23",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.2",
            "lineages": ["BA.2", "BA.2.1", "BA.2.10", "BA.2.13", "BA.2.18", "BA.2.21", "BA.2.23", "BA.2.26", "BA.2.3", "BA.2.37", "BA.2.48", "BA.2.65", "BA.2.7", "BA.2.9"],
            "start_date": "2022-04-23",
            "end_date": "2022-06-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.4",
            "lineages": ["BA.4", "BA.4.1", "BA.4.2", "BA.4.4", "BA.4.6"],
            "start_date": "2022-04-23",
            "end_date": "2022-06-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5",
            "lineages": ["BA.5", "BA.5.1", "BA.5.2", "BA.5.5", "BA.5.6"],
            "start_date": "2022-04-23",
            "end_date": "2022-06-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.1",
            "lineages": ["BA.5.1", "BA.5.1.1", "BA.5.1.10", "BA.5.1.2", "BA.5.1.22", "BA.5.1.23", "BA.5.1.24", "BA.5.1.25", "BA.5.1.3", "BA.5.1.30", "BA.5.1.5", "BA.5.1.6"],
            "start_date": "2022-06-22",
            "end_date": "2022-08-21",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.1",
            "lineages": ["BA.5.1", "BA.5.1.1", "BA.5.1.10", "BA.5.1.18", "BA.5.1.2", "BA.5.1.22", "BA.5.1.23", "BA.5.1.24", "BA.5.1.25", "BA.5.1.27", "BA.5.1.3", "BA.5.1.30", "BA.5.1.5", "BA.5.1.6"],
            "start_date": "2022-08-21",
            "end_date": "2022-10-20",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2",
            "lineages": ["BA.5.2", "BA.5.2.20", "BA.5.2.21", "BA.5.2.22", "BA.5.2.3", "BA.5.2.31", "BA.5.2.9"],
            "start_date": "2022-06-22",
            "end_date": "2022-08-21",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2",
            "lineages": ["BA.5.2", "BA.5.2.20", "BA.5.2.21", "BA.5.2.22", "BA.5.2.23", "BA.5.2.3", "BA.5.2.31", "BA.5.2.34", "BA.5.2.6", "BA.5.2.9"],
            "start_date": "2022-08-21",
            "end_date": "2022-10-20",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2",
            "lineages": ["BA.5.2", "BA.5.2.1", "BA.5.2.20", "BA.5.2.21", "BA.5.2.23", "BA.5.2.34", "BA.5.2.6", "BA.5.2.9"],
            "start_date": "2022-10-20",
            "end_date": "2022-12-19",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2.1",
            "lineages": ["BA.5.2.1", "BF.1", "BF.10", "BF.21", "BF.27", "BF.28", "BF.5", "BF.8"],
            "start_date": "2022-04-23",
            "end_date": "2022-06-22",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2.1",
            "lineages": ["BA.5.2.1", "BF.1", "BF.10", "BF.13", "BF.21", "BF.26", "BF.27", "BF.28", "BF.4", "BF.5", "BF.7", "BF.8", "BF.9"],
            "start_date": "2022-06-22",
            "end_date": "2022-08-21",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2.1",
            "lineages": ["BA.5.2.1", "BF.1", "BF.10", "BF.11", "BF.13", "BF.14", "BF.21", "BF.26", "BF.27", "BF.28", "BF.4", "BF.5", "BF.7", "BF.8", "BF.9"],
            "start_date": "2022-08-21",
            "end_date": "2022-10-20",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BA.5.2.1",
            "lineages": ["BA.5.2.1", "BF.1", "BF.10", "BF.11", "BF.13", "BF.14", "BF.21", "BF.26", "BF.27", "BF.28", "BF.4", "BF.5", "BF.7", "BF.8", "BF.9"],
            "start_date": "2022-10-20",
            "end_date": "2022-12-19",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BQ.1",
            "lineages": ["BQ.1", "BQ.1.1", "BQ.1.2", "BQ.1.10", "BQ.1.11", "BQ.1.12", "BQ.1.13", "BQ.1.14", "BQ.1.23", "BQ.1.32", "BQ.1.5"],
            "start_date": "2022-10-20",
            "end_date": "2022-12-19",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BQ.1",
            "lineages": ["BQ.1", "BQ.1.1", "BQ.1.2", "BQ.1.3", "BQ.1.5", "BQ.1.10", "BQ.1.11", "BQ.1.12", "BQ.1.13", "BQ.1.14", "BQ.1.23", "BQ.1.32"],
            "start_date": "2022-12-19",
            "end_date": "2023-02-17",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BQ.1.1",
            "lineages": ["BQ.1.1", "BQ.1.1.1", "BQ.1.1.18", "BQ.1.1.3", "BQ.1.1.32", "BQ.1.1.4", "BQ.1.1.41", "BQ.1.1.5", "BQ.1.1.51", "BQ.1.1.68", "BQ.1.1.69", "BQ.1.1.7"],
            "start_date": "2022-10-20",
            "end_date": "2022-12-19",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "BQ.1.1",
            "lineages": ["BQ.1.1", "BQ.1.1.1", "BQ.1.1.18", "BQ.1.1.3", "BQ.1.1.32", "BQ.1.1.4", "BQ.1.1.41", "BQ.1.1.5", "BQ.1.1.51", "BQ.1.1.68", "BQ.1.1.69", "BQ.1.1.7"],
            "start_date": "2022-12-19",
            "end_date": "2023-02-17",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "EG.5.1",
            "lineages": ["EG.5.1", "EG.5.1.1", "EG.5.1.16", "EG.5.1.3", "EG.5.1.4", "EG.5.1.6"],
            "start_date": "2023-06-17",
            "end_date": "2023-08-16",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "EG.5.1",
            "lineages": ["EG.5.1", "EG.5.1.1", "EG.5.1.16", "EG.5.1.3", "EG.5.1.4", "EG.5.1.6"],
            "start_date": "2023-08-16",
            "end_date": "2023-10-15",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "EG.5.1",
            "lineages": ["EG.5.1", "EG.5.1.1", "EG.5.1.16", "EG.5.1.3", "EG.5.1.4", "EG.5.1.6"],
            "start_date": "2023-10-15",
            "end_date": "2023-12-14",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "JN.1",
            "lineages": ["JN.1", "JN.1.1", "JN.1.2", "JN.1.39", "JN.1.4", "JN.1.42", "JN.1.7", "JN.1.9"],
            "start_date": "2023-12-14",
            "end_date": "2024-02-12",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "JN.1",
            "lineages": ["JN.1", "JN.1.1", "JN.1.39", "JN.1.4", "JN.1.42", "JN.1.7"],
            "start_date": "2024-02-12",
            "end_date": "2024-04-12",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "KP.3.1.1",
            "lineages": ["KP.3.1.1", "MC.1", "MC.13", "MC.16", "MC.24"],
            "start_date": "2024-08-10",
            "end_date": "2024-10-09",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "KP.3.1.1",
            "lineages": ["KP.3.1.1", "MC.1", "MC.13", "MC.16", "MC.24"],
            "start_date": "2024-10-09",
            "end_date": "2024-12-08",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "XBB.1.5",
            "lineages": ["XBB.1.5", "XBB.1.5.1", "XBB.1.5.11", "XBB.1.5.13", "XBB.1.5.15", "XBB.1.5.16", "XBB.1.5.17", "XBB.1.5.19", "XBB.1.5.20", "XBB.1.5.21", "XBB.1.5.31", "XBB.1.5.32", "XBB.1.5.33", "XBB.1.5.4", "XBB.1.5.49", "XBB.1.5.51", "XBB.1.5.67"],
            "start_date": "2022-12-19",
            "end_date": "2023-02-17",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "XBB.1.5",
            "lineages": ["XBB.1.5", "XBB.1.5.1", "XBB.1.5.10", "XBB.1.5.11", "XBB.1.5.13", "XBB.1.5.15", "XBB.1.5.16", "XBB.1.5.17", "XBB.1.5.19", "XBB.1.5.20", "XBB.1.5.21", "XBB.1.5.31", "XBB.1.5.32", "XBB.1.5.33", "XBB.1.5.35", "XBB.1.5.4", "XBB.1.5.49", "XBB.1.5.51", "XBB.1.5.67"],
            "start_date": "2023-02-17",
            "end_date": "2023-04-18",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "XBB.1.5",
            "lineages": ["XBB.1.5", "XBB.1.5.1", "XBB.1.5.10", "XBB.1.5.13", "XBB.1.5.15", "XBB.1.5.17", "XBB.1.5.35", "XBB.1.5.4", "XBB.1.5.49"],
            "start_date": "2023-04-18",
            "end_date": "2023-06-17",
            "protein": "surface glycoprotein"
        },
        {
            "parent_lineage": "XBB.1.16",
            "lineages": ["XBB.1.16", "XBB.1.16.1", "XBB.1.16.6", "XBB.1.16.11", "XBB.1.16.15"],
            "start_date": "2023-08-16",
            "end_date": "2023-10-15",
            "protein": "surface glycoprotein"
        }
    ]
    
    try:
        # Process each parameter set (no model loading needed for Levenshtein)
        all_results = []
        for params in parameter_sets:
            try:
                result = analyze_lineage_fitness(**params)
                if result is not None and not result.empty:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing {params['parent_lineage']}: {e}")
                continue
        
        # Optionally combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_results.to_csv("combined_lineage_analysis_levenshtein.csv", index=False)
            print(f"Processed {len(parameter_sets)} parameter sets successfully")
    
    except Exception as e:
        print(f"Error in main execution: {e}")