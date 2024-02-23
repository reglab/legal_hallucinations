import json
import random

import pandas as pd
from pandas.core.frame import DataFrame

from settings import (
    F1D_PATH,
    F1SUPP_PATH,
    F2D_PATH,
    F2SUPP_PATH,
    F3D_PATH,
    F3SUPP_PATH,
    FD_SAMPLE_PATH,
    FOWLER_SCORES_PATH,
    FSUPP_SAMPLE_PATH,
    RANDOM_SEED,
    SCDB_LEGACY_PATH,
    SCDB_PATH,
    SCDB_SAMPLE_PATH,
    SCOTUS_MAJORITY_OPINIONS_DIRECTORY,
    SCOTUS_SHEPARDS_SAMPLE,
    SHEPARDS_PATH,
    SONGER_DB_PATH,
    SONGER_SAMPLE_PATH,
)
from utils import (
    get_circuit_from_cap_id,
    get_disposition_from_scdb_id,
    get_disposition_from_songer_id,
    get_majority_author_from_cap_dict,
    get_state_from_cap_slug,
)

random.seed(RANDOM_SEED)

###################################
# Generate SCOTUS data
###################################
SCDB: DataFrame = pd.read_csv(SCDB_PATH, index_col=False, encoding="ISO-8859-1")
SCDB_LEGACY: DataFrame = pd.read_csv(
    SCDB_LEGACY_PATH, index_col=False, encoding="ISO-8859-1"
)
SCDB = pd.concat([SCDB, SCDB_LEGACY]).reset_index().sort_values("caseId")
FOWLER_SCORES: DataFrame = pd.read_csv(FOWLER_SCORES_PATH, index_col=False)
SCDB = SCDB.merge(FOWLER_SCORES, how="left", left_on="lexisCite", right_on="lex_id")

# Filter data
scdb_sample: DataFrame = SCDB[
    SCDB.majOpinWriter.notnull()
    & SCDB.caseDisposition.notnull()
    & SCDB.issueArea.notnull()
    & SCDB.usCite.notnull()
    & SCDB.caseName.notnull()
].copy()
scdb_sample.term = scdb_sample.term.astype("int")
scdb_sample.issueArea = scdb_sample.issueArea.astype("int")
scdb_sample.majOpinWriter = scdb_sample.majOpinWriter.astype("int")
scdb_sample.caseDisposition = scdb_sample.caseDisposition.astype("int")
scdb_sample.partyWinning = scdb_sample.partyWinning.astype("int")
scdb_sample["disposition"] = scdb_sample.caseDisposition.apply(
    get_disposition_from_scdb_id
)  # Create coarsened disposition column
scdb_sample = scdb_sample[scdb_sample.partyWinning != 2]  # Filter out unclear winners
scdb_sample = scdb_sample[
    scdb_sample.disposition != ""
]  # Filter out unclear dispositions

# Sample from strata
scdb_sample = scdb_sample.groupby("term").apply(
    lambda x: x.sample(
        len(x) if len(x) < 25 else 25, replace=False, random_state=RANDOM_SEED
    )
)  # Sample uniformly conditional on year


# Merge with majority opinion content
def get_majority_opinion(file_name: str):
    try:
        with open(SCOTUS_MAJORITY_OPINIONS_DIRECTORY + file_name + ".txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""


scdb_sample["majority_opinion"] = scdb_sample["lexisCite"].apply(get_majority_opinion)
scdb_sample = scdb_sample[
    scdb_sample.majority_opinion != ""
]  # Filter out cases without a majority opinion
scdb_sample = scdb_sample.sample(
    5000, replace=False, random_state=RANDOM_SEED
)  # Downsample to 5000

# Save to disk
scdb_sample.to_csv(SCDB_SAMPLE_PATH, index=False)

###################################
# Generate COA data (CAP)
###################################
files: dict = {
    "supp1": {"path": F1D_PATH, "cases": 72368},
    "supp2": {"path": F2D_PATH, "cases": 275122},
    "supp3": {"path": F3D_PATH, "cases": 141483},
}

coa_sample: list = []
for v in files.values():
    sample_lines: list[int] = random.sample(
        range(v["cases"]), int(v["cases"] * 0.2)
    )  # Only load a part of the data into memory
    sample_lines.sort()

    with open(v["path"], "r") as f:
        for i, line in enumerate(f):
            if i == sample_lines[0]:
                coa_sample.append(json.loads(line))
                sample_lines.pop(0)

                if not sample_lines:
                    break

# Limit to only post-1895 cases (very few earlier ones are published in the F. reporter.)
coa_sample_df: DataFrame = pd.DataFrame(coa_sample)
coa_sample_df["year"] = coa_sample_df.decision_date.str[0:4].astype(int)
coa_sample_df = coa_sample_df[coa_sample_df.year >= 1895]

# Filter out cases with missing authors
coa_sample_df["majority_author"] = coa_sample_df["casebody"].apply(
    lambda x: get_majority_author_from_cap_dict(x)
)
coa_sample_df = coa_sample_df[coa_sample_df.majority_author.notnull()]

# Create year-circuit strata
coa_sample_df["circuit"] = coa_sample_df["court"].apply(
    lambda x: get_circuit_from_cap_id(x["id"])
)
coa_sample_df = coa_sample_df[coa_sample_df["circuit"] != 99]
coa_sample_df["stratum"] = (
    coa_sample_df["circuit"].astype(str) + "_" + coa_sample_df["year"].astype(str)
)

# Sample from strata
coa_sample_df = coa_sample_df.groupby("stratum").apply(
    lambda x: x.sample(
        len(x) if len(x) < 4 else 4, replace=False, random_state=RANDOM_SEED
    )
)
coa_sample_df = coa_sample_df.sample(
    5000, replace=False, random_state=RANDOM_SEED
)  # Downsample to 5000
coa_sample_df.reset_index(drop=True, inplace=True)
coa_sample_df.to_csv(FD_SAMPLE_PATH, index=False)

###################################
# Generate COA data (Songer)
###################################
coa_sample_df2: DataFrame = pd.read_csv(SONGER_DB_PATH, index_col=False)

# Filter out cases with missing names
coa_sample_df2 = coa_sample_df2[coa_sample_df2.case_name.notnull()]

# Create year-circuit strata
coa_sample_df2.loc[coa_sample_df2["circuit"] == 0, "circuit"] = 13  # 0 == DC circuit
coa_sample_df2["stratum"] = (
    coa_sample_df2["circuit"].astype(str) + "_" + coa_sample_df2["year"].astype(str)
)

# Filter out cases with unclear dispositions
coa_sample_df2["disposition"] = coa_sample_df2["treat"].apply(
    get_disposition_from_songer_id
)
coa_sample_df2 = coa_sample_df2[coa_sample_df2.disposition.notnull()]
coa_sample_df2["disposition"] = coa_sample_df2["disposition"].astype(int)

# Sample from strata
coa_sample_df2 = coa_sample_df2.groupby("stratum").apply(
    lambda x: x.sample(
        len(x) if len(x) < 6 else 6, replace=False, random_state=RANDOM_SEED
    )
)
coa_sample_df2 = coa_sample_df2.sample(
    5000, replace=False, random_state=RANDOM_SEED
)  # Downsample to 5000
coa_sample_df2.reset_index(drop=True, inplace=True)
coa_sample_df2.to_csv(SONGER_SAMPLE_PATH, index=False)

###################################
# Generate USDC data
###################################
files = {
    "supp1": {"path": F1SUPP_PATH, "cases": 214096},
    "supp2": {"path": F2SUPP_PATH, "cases": 115461},
    "supp3": {"path": F3SUPP_PATH, "cases": 36685},
}

usdc_sample: list = []
for v in files.values():
    with open(v["path"], "r") as f:
        for i, line in enumerate(f):
            usdc_sample.append(json.loads(line))

# Limit to only post-1932 cases (very few earlier ones are published in the F. Supp.)
usdc_sample_df: DataFrame = pd.DataFrame(usdc_sample)
usdc_sample_df["year"] = usdc_sample_df.decision_date.str[0:4].astype(int)
usdc_sample_df = usdc_sample_df[usdc_sample_df.year >= 1932]

# Filter out cases with missing authors
usdc_sample_df["majority_author"] = usdc_sample_df["casebody"].apply(
    lambda x: get_majority_author_from_cap_dict(x)
)
usdc_sample_df = usdc_sample_df[usdc_sample_df.majority_author.notnull()]

# Create year-state strata
usdc_sample_df["state"] = usdc_sample_df["court"].apply(
    lambda x: get_state_from_cap_slug(x["slug"])
)
usdc_sample_df = usdc_sample_df[usdc_sample_df["state"] != "misc"]
usdc_sample_df["stratum"] = (
    usdc_sample_df["state"].astype(str) + "_" + usdc_sample_df["year"].astype(str)
)

# Sample from strata
usdc_sample_df = usdc_sample_df.groupby("stratum").apply(
    lambda x: x.sample(
        len(x) if len(x) < 2 else 2, replace=False, random_state=RANDOM_SEED
    )
)  # 50 * (2021-1932) ~= 8900
usdc_sample_df = usdc_sample_df.sample(
    5000, replace=False, random_state=RANDOM_SEED
)  # Downsample to 5000
usdc_sample_df.reset_index(drop=True, inplace=True)
usdc_sample_df.to_csv(FSUPP_SAMPLE_PATH, index=False)

###################################
# Generate SCOTUS Shepard's data
###################################
SHEPARDS_DB: DataFrame = pd.read_csv(SHEPARDS_PATH, index_col=False)

# Filter to SCOTUS data and merge with SCDB
scotus_shepards_sample: DataFrame = SHEPARDS_DB[SHEPARDS_DB.supreme_court == 1]
scotus_shepards_sample = (
    scotus_shepards_sample.merge(
        SCDB[["lex_id", "caseName", "usCite"]],
        how="left",
        left_on="citing_case",
        right_on="lex_id",
    )
    .drop("lex_id", axis=1)
    .rename(columns={"caseName": "citing_case_name", "usCite": "citing_case_us_cite"})
)
scotus_shepards_sample = (
    scotus_shepards_sample.merge(
        SCDB[["lex_id", "caseName", "usCite"]],
        how="left",
        left_on="cited_case",
        right_on="lex_id",
    )
    .drop("lex_id", axis=1)
    .rename(columns={"caseName": "cited_case_name", "usCite": "cited_case_us_cite"})
)
scotus_shepards_sample = scotus_shepards_sample.dropna(subset=["citing_case_name"])
scotus_shepards_sample = scotus_shepards_sample.dropna(subset=["cited_case_name"])

# Take sample of data
scotus_shepards_sample = scotus_shepards_sample[
    scotus_shepards_sample.shepards.isin(
        [
            "distinguished",
            "criticized",
            "limit",
            "questioned",
            "overrul",
            "followed",
            "parallel",
        ]
    )
]
scotus_shepards_sample = scotus_shepards_sample.drop_duplicates(
    subset=["citing_case"]
)  # Filter out duplicate citing cases to avoid clustering of hallucinations
scotus_shepards_sample = scotus_shepards_sample.groupby("citing_case_year").apply(
    lambda x: x.sample(
        len(x) if len(x) < 35 else 35, replace=False, random_state=RANDOM_SEED
    )
)  # Sample uniformly conditional on year
scotus_shepards_sample.citing_case_year = (
    scotus_shepards_sample.citing_case_year.astype("int")
)
scotus_shepards_sample.cited_case_year = scotus_shepards_sample.cited_case_year.astype(
    "int"
)
scotus_shepards_sample = scotus_shepards_sample.sample(
    5000, replace=False, random_state=RANDOM_SEED
)  # Downsample to match total sample size of normal SCOTUS data

# Coarsen Shepard's codes into binary positive/negative
scotus_shepards_sample["agree"] = scotus_shepards_sample.shepards.isin(
    ["followed", "parallel"]
).astype("int")

# Save to disk
scotus_shepards_sample.to_csv(SCOTUS_SHEPARDS_SAMPLE, index=False)
