import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import linregress, t
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ============================================================
# STEP 1 — LOAD AND COMBINE ALL CSVs WITH GROUP IDENTIFIER
# ============================================================

folder_path = "/home/ez1199/levsummary_test"
distance = "lev"
output_dir = "/home/ez1199/test/covariance_levenshtein"
os.makedirs(output_dir, exist_ok=True)
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        df.columns = ["Parent Lineage", "Child Lineage", "Levenshtein Distance", "Fitness Difference", "Fitness Lower", "Fitness Upper"]

        # Expected format: summary_PARENT_STARTtoEND_surface_glycoprotein_levenshtein.csv
        try:
            base = filename.replace("summary_", "").replace("_surface_glycoprotein_levenshtein.csv", "")
            *parent_parts, date_part = base.split("_")
            parent_lineage = "_".join(parent_parts)
            start_date, end_date = date_part.split("to")
        except Exception as e:
            print(f"Warning: Could not parse dates from {filename}: {e}")
            start_date, end_date = "unknown", "unknown"
            parent_lineage = "unknown"

        df["Group"] = f"{parent_lineage}_{start_date}_to_{end_date}"
        df["Start Date"] = start_date
        df["End Date"] = end_date

        all_data.append(df)

if not all_data:
    print("No CSV files found in folder. Exiting.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Clean numeric columns
combined_df["Levenshtein Distance"] = pd.to_numeric(combined_df["Levenshtein Distance"], errors="coerce")
combined_df["Fitness Difference"] = pd.to_numeric(combined_df["Fitness Difference"], errors="coerce")
combined_df = combined_df.dropna(subset=["Levenshtein Distance", "Fitness Difference"])
combined_df = combined_df[
    np.isfinite(combined_df["Levenshtein Distance"]) & np.isfinite(combined_df["Fitness Difference"])
]

print(f"Total data points loaded: {len(combined_df)}")
print(f"Total groups found: {combined_df['Group'].nunique()}")
print(f"Groups found:\n{combined_df['Group'].unique()}")

# Check group sizes
group_sizes = combined_df.groupby("Group").size()
print(f"\nGroups with fewer than 4 points:")
print(group_sizes[group_sizes < 4])
print(f"\nAll group sizes:")
print(group_sizes.sort_values())

# ============================================================
# STEP 2 — RUN ANALYSIS FOR EACH THRESHOLD
# ============================================================

thresholds = [3, 5, 8]  # 3 = minimum valid, 5 = moderate filter, 8 = conservative filter

for min_n in thresholds:
    print(f"\n{'='*60}")
    print(f"ANALYSIS WITH MINIMUM GROUP SIZE: {min_n}")
    print(f"{'='*60}")

    # Filter groups
    valid_groups = group_sizes[group_sizes >= min_n].index.tolist()
    filtered_df = combined_df[combined_df["Group"].isin(valid_groups)].copy()

    print(f"Groups included:      {filtered_df['Group'].nunique()}")
    print(f"Groups excluded:      {combined_df['Group'].nunique() - filtered_df['Group'].nunique()}")
    print(f"Total observations:   {len(filtered_df)}")

    # ---- ANCOVA ----
    ancova_df = filtered_df.rename(columns={
        "Levenshtein Distance": "Lev_Dist",
        "Fitness Difference": "Fit_Diff"
    })

    ancova_model = smf.ols(
        "Fit_Diff ~ Lev_Dist * C(Group, Sum)",
        data=ancova_df
    ).fit()

    print(f"R-squared:            {ancova_model.rsquared:.3f}")
    print(f"Adjusted R-squared:   {ancova_model.rsquared_adj:.3f}")
    print(f"F-statistic p-value:  {ancova_model.f_pvalue:.4f}")

    summary_table = ancova_model.summary2().tables[1]
    Lev_terms = summary_table.filter(like="Lev_Dist", axis=0)
    print("\nLevenshtein Distance Coefficients:")
    print(Lev_terms.to_string())

    # Save ANCOVA results
    ancova_summary = pd.DataFrame({
        "Metric": ["Min Group Size", "Total Observations", "Number of Groups",
                   "R-squared", "Adjusted R-squared", "F-statistic p-value",
                   "Grand Mean Slope", "Grand Mean Slope p-value"],
        "Value": [min_n, int(ancova_model.nobs), filtered_df["Group"].nunique(),
                  round(ancova_model.rsquared, 3), round(ancova_model.rsquared_adj, 3),
                  round(ancova_model.f_pvalue, 4),
                  round(Lev_terms["Coef."].iloc[0], 4),
                  round(Lev_terms["P>|t|"].iloc[0], 4)]
    })
    ancova_summary.to_csv(os.path.join(output_dir, f"{distance}_ancova_summary_n{min_n}.csv"), index=False)
    Lev_terms.to_csv(os.path.join(output_dir, f"{distance}_ancova_results_n{min_n}.csv"))
    print(f"Saved {distance}_ancova_summary_n{min_n}.csv and {distance}_ancova_results_n{min_n}.csv")

    # ---- Per-group slopes ----
    print(f"\n{'-'*40}")
    print("PER-GROUP SLOPE ANALYSIS")
    print(f"{'-'*40}")

    slope_results = []

    for group, subset in filtered_df.groupby("Group"):
        x = subset["Levenshtein Distance"].to_numpy()
        y = subset["Fitness Difference"].to_numpy()
        n = len(x)

        if n < 3:
            print(f"Skipping {group}: insufficient observations (n={n})")
            continue

        if len(np.unique(x)) < 2:
            print(f"Skipping {group}: all Levenshtein distances are identical")
            continue

        slope, intercept, r, p, se = linregress(x, y)
        t_crit = t.ppf(0.975, df=n - 2)
        ci = t_crit * se

        slope_results.append({
            "Group": group,
            "Parent Lineage": subset["Parent Lineage"].iloc[0],
            "Start Date": subset["Start Date"].iloc[0],
            "End Date": subset["End Date"].iloc[0],
            "Slope": slope,
            "CI": ci,
            "R": r,
            "P-value": p,
            "N": n,
            "Significant": p < 0.05,
            "Direction": "Positive" if slope > 0 else "Negative"
        })

    slopes_df = pd.DataFrame(slope_results).sort_values("Start Date", ascending=False)

    # Apply Bonferroni correction across all per-group p-values
    reject, p_adj, _, _ = multipletests(slopes_df["P-value"].values, alpha=0.05, method="bonferroni")
    slopes_df["P-value (Bonferroni adjusted)"] = p_adj
    slopes_df["Significant (Bonferroni)"] = reject

    slopes_df.to_csv(os.path.join(output_dir, f"{distance}_slopes_by_group_n{min_n}.csv"), index=False)
    print(f"Saved {distance}_slopes_by_group_n{min_n}.csv")

    print(f"Positive slopes:      {(slopes_df['Slope'] > 0).sum()} / {len(slopes_df)}")
    print(f"Negative slopes:      {(slopes_df['Slope'] < 0).sum()} / {len(slopes_df)}")
    print(f"Significant (p<0.05, uncorrected): {slopes_df['Significant'].sum()} / {len(slopes_df)}")
    print(f"Significant (Bonferroni):          {slopes_df['Significant (Bonferroni)'].sum()} / {len(slopes_df)}")
    print("\nFull slope table:")
    print(slopes_df[["Group", "Slope", "CI", "P-value", "P-value (Bonferroni adjusted)", "N", "Direction"]].to_string(index=False))

    # ---- Stratified slope plot ----
    unique_lineages = slopes_df.iloc[::-1]["Parent Lineage"].unique()
    colors = plt.cm.tab20.colors
    color_map = {lin: colors[i % len(colors)] for i, lin in enumerate(unique_lineages)}

    def fmt_label(start):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        s = pd.Timestamp(start)
        return f"{months[s.month-1]} {s.year}"

    labels = [fmt_label(row["Start Date"]) for _, row in slopes_df.iterrows()]

    n_groups = len(slopes_df)
    fig, ax = plt.subplots(figsize=(6.5, 9))

    for i, (_, row) in enumerate(slopes_df.iterrows()):
        color = color_map[row["Parent Lineage"]]
        ax.errorbar(
            x=row["Slope"], y=i, xerr=row["CI"],
            fmt="o", color=color, capsize=4, linewidth=1.5,
            markersize=7, markeredgecolor="black", markeredgewidth=0.5
        )

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    handles = [
        matplotlib.patches.Patch(color=color_map[lin], label=lin)
        for lin in unique_lineages
    ]
    ax.legend(handles=handles, title="Parent Lineage",
              bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax.set_xlabel("Slope (95% CI)", fontsize=11)
    ax.set_ylabel("Time Window Start Date", fontsize=11)
    ax.set_title(f"Per-Group Regression Slopes (min n={min_n})\nLog Relative Fitness Change ~ Levenshtein Distance",
                 fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{distance}_slopes_by_group_n{min_n}.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {distance}_slopes_by_group_n{min_n}.png")