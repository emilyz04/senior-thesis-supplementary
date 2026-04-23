import os
import time
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# Accession prefixes whose GenBank entries typically carry full CDS annotations.
# Prioritise these to avoid wasting calls on unannotated ENA/raw submissions.
ANNOTATED_PREFIXES = ("MW", "MZ", "OK", "OL", "OM", "ON", "OP", "OQ", "OR",
                      "OV", "OX", "PP", "PQ", "PV")

def _priority_sort(accessions):
    """Return accessions sorted so likely-annotated prefixes come first."""
    priority, rest = [], []
    for a in accessions:
        (priority if str(a)[:2] in ANNOTATED_PREFIXES else rest).append(a)
    return priority + rest

METADATA_PATH = "/scratch/gpfs/GRENFELL/ez1199/public-latest.metadata.tsv"
BASE_SCRATCH  = "/scratch/gpfs/GRENFELL/ez1199/test_data"

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


def try_download_spike(accession, output_path, protein):
    """Try to download spike protein FASTA for one GenBank accession.
    Returns True on success, False on any failure."""
    acc_versioned = accession if "." in str(accession) else f"{accession}.1"
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/virus/accession/{acc_versioned}/annotation_report"
    try:
        time.sleep(0.4)   # stay under NCBI's 3 req/sec limit
        for attempt in range(5):
            response = requests.get(url, timeout=30)
            if response.status_code != 429:
                break
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            print(f"    [429] rate-limited, backing off {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
        report = response.json()
        reports = report.get("reports", [])
        if not reports:
            return False

        protein_accession = None
        for gene in reports[0].get("genes", []):
            for cds in gene.get("cds", []):
                name    = cds.get("name", "").lower()
                product = cds.get("product", "").lower()
                others  = [n.lower() for n in cds.get("other_names", [])]
                if (protein.lower() in name or protein.lower() in product
                        or any(protein.lower() in n for n in others)):
                    protein_accession = cds.get("protein", {}).get("accession_version")
                    break
            if protein_accession:
                break

        if not protein_accession:
            return False

        # Fetch protein FASTA directly via NCBI efetch (no external tool needed)
        time.sleep(0.4)
        efetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=protein&id={protein_accession}&rettype=fasta&retmode=text"
            f"&email=jorrel@princeton.edu"
        )
        fa_response = requests.get(efetch_url, timeout=30)
        if fa_response.status_code != 200 or not fa_response.text.startswith(">"):
            return False

        lines = fa_response.text.strip().splitlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))

        if not seq or "X" in seq.upper():
            return False

        with open(output_path, "w") as f:
            f.write(fa_response.text)
        return True

    except Exception as e:
        print(f"    [err] {accession}: {e}")
        return False


def main():
    print(f"Loading {METADATA_PATH} ...")
    df = pd.read_csv(
        METADATA_PATH, sep='\t', low_memory=False,
        usecols=["genbank_accession", "pango_lineage_usher", "pangolin_lineage", "country", "date"]
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"Loaded {len(df):,} rows")

    missing_total = 0
    downloaded    = 0
    still_missing = 0

    for params in parameter_sets:
        parent_lineage = params["parent_lineage"]
        lineages       = params["lineages"]
        start_date     = params["start_date"]
        end_date       = params["end_date"]
        protein        = params["protein"]
        protein_dir    = protein.replace(" ", "_")

        output_dir = f"{BASE_SCRATCH}/{parent_lineage}_{protein_dir}_{start_date}to{end_date}/"
        os.makedirs(output_dir, exist_ok=True)

        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)

        for lineage in lineages:
            fa_path = os.path.join(output_dir, f"{lineage}.fa")
            if os.path.exists(fa_path):
                print("Skipping: Already Have", fa_path)
                continue  # already have it

            missing_total += 1
            print(f"\n[MISSING] {lineage} in {parent_lineage} {start_date}→{end_date}")

            # Filter cascade: strict → relaxed
            # Use pango_lineage_usher primarily, fall back to pangolin_lineage
            filter_cascade = [
                ("date+country+usher",   lambda d, lin: d[(d["pango_lineage_usher"] == lin) & (d["country"] == "USA") & (d["date"] >= start_dt) & (d["date"] <= end_dt)]),
                ("date+country+pango",   lambda d, lin: d[(d["pangolin_lineage"]    == lin) & (d["country"] == "USA") & (d["date"] >= start_dt) & (d["date"] <= end_dt)]),
                ("nodate+country+usher", lambda d, lin: d[(d["pango_lineage_usher"] == lin) & (d["country"] == "USA")]),
                ("nodate+country+pango", lambda d, lin: d[(d["pangolin_lineage"]    == lin) & (d["country"] == "USA")]),
                ("global+usher",         lambda d, lin: d[ d["pango_lineage_usher"] == lin]),
                ("global+pango",         lambda d, lin: d[ d["pangolin_lineage"]    == lin]),
            ]

            success = False
            for desc, filt_fn in filter_cascade:
                candidates = filt_fn(df, lineage)
                accessions = _priority_sort(candidates["genbank_accession"].dropna().unique().tolist())
                if not accessions:
                    continue
                print(f"  Trying {desc}: {len(accessions)} accessions (top prefix: {accessions[0][:2] if accessions else '?'})")
                for acc in accessions[:20]:   # cap at 20 tries per filter level
                    if try_download_spike(acc, fa_path, protein):
                        print(f"  ✓ Downloaded via {desc} (accession {acc})")
                        downloaded += 1
                        success = True
                        break
                if success:
                    break

            if not success:
                print(f"  ✗ No valid spike found for {lineage}")
                still_missing += 1

    print(f"\n{'='*60}")
    print(f"Total missing:    {missing_total}")
    print(f"Downloaded:       {downloaded}")
    print(f"Still missing:    {still_missing}")


if __name__ == "__main__":
    main()



