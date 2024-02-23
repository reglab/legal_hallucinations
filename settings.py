import os

BASE_DIRECTORY = "/Users/mattdahl/Documents/reglab/projects/legal_hallucinations"

# Data sources
SCDB_PATH: str = os.path.join(
    BASE_DIRECTORY, "data/sources/SCDB_2022_01_caseCentered_Citation.csv"
)
SCDB_LEGACY_PATH: str = os.path.join(
    BASE_DIRECTORY, "data/sources/SCDB_Legacy_07_caseCentered_Citation.csv"
)
FOWLER_SCORES_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/fowler_scores.csv")
F1SUPP_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/cap_f1supp_text.jsonl")
F2SUPP_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/cap_f2supp_text.jsonl")
F3SUPP_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/cap_f3supp_text.jsonl")
F1D_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/cap_f1d_text.jsonl")
F2D_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/cap_f2d_text.jsonl")
F3D_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/cap_f3d_text.jsonl")
SHEPARDS_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/shepards_data.csv")
SCOTUS_MAJORITY_OPINIONS_DIRECTORY: str = os.path.join(
    BASE_DIRECTORY, "data/sources/scotus_majority_opinions/"
)
SONGER_DB_PATH: str = os.path.join(BASE_DIRECTORY, "data/sources/songer_db.csv")

# Data samples
SCDB_SAMPLE_PATH: str = os.path.join(BASE_DIRECTORY, "data/samples/scdb_sample.csv")
FSUPP_SAMPLE_PATH: str = os.path.join(BASE_DIRECTORY, "data/samples/fsupp_sample.csv")
FD_SAMPLE_PATH: str = os.path.join(BASE_DIRECTORY, "data/samples/fd_sample.csv")
SONGER_SAMPLE_PATH: str = os.path.join(BASE_DIRECTORY, "data/samples/songer_sample.csv")
SCOTUS_SHEPARDS_SAMPLE: str = os.path.join(
    BASE_DIRECTORY, "data/samples/scotus_shepards_sample.csv"
)
FAKE_CASES_DB: str = os.path.join(BASE_DIRECTORY, "data/samples/fake_cases.csv")
SCOTUS_OVERRULED_DB: str = os.path.join(
    BASE_DIRECTORY, "data/sources/scotus_overruled_db.csv"
)

# Covariates
SCDB_JUSTICE_MAPPING_PATH: str = os.path.join(
    BASE_DIRECTORY, "data/covariates/scdb_justice_name_map.csv"
)

# Results
RESULTS_SAVE_PATH: str = os.path.join(BASE_DIRECTORY, "results/tasks")
FIGURES_SAVE_PATH: str = os.path.join(BASE_DIRECTORY, "results/figures")
OBJECTS_SAVE_PATH: str = os.path.join(BASE_DIRECTORY, "results/objects")

# Threading
NUM_THREADS = 10

# Misc
RANDOM_SEED: int = 47
