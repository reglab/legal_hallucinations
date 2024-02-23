import argparse
import random
from typing import cast

import pandas as pd
from pandas.core.frame import DataFrame

from api import (
    GooglePaLMCompletion,
    LlamaChat,
    OpenAIChat,
    OpenAIChatGpt4,
    TogetherAiLlamaChat,
)
from correctness_checks import (
    bool_correctness,
    clean_judge_name,
    fake_dissent_correctness,
    fake_overruling_correctness,
)
from models import CourtCase, Query, Task
from settings import (
    FAKE_CASES_DB,
    FD_SAMPLE_PATH,
    RANDOM_SEED,
    SCDB_SAMPLE_PATH,
    SCOTUS_OVERRULED_DB,
)
from utils import (
    APIBackendType,
    format_case_name,
    get_case_citation_for_scotus_case,
    get_citation_from_cap_dict,
    get_importance_from_cap_dict,
    get_judge_name_from_scdb_id,
    noop,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--api", type=str, help="api to use", choices=["llama", "gpt3.5", "palm", "gpt4"]
)
args = parser.parse_args()

CURRENT_API: APIBackendType = OpenAIChatGpt4
match args.api:
    case "llama":
        CURRENT_API = LlamaChat  # TogetherAiLlamaChat also okay
    case "gpt3.5":
        CURRENT_API = OpenAIChat
    case "palm":
        CURRENT_API = GooglePaLMCompletion
    case "gpt4":
        CURRENT_API = OpenAIChatGpt4

random.seed(RANDOM_SEED)

###################################
#  Fake existence task
###################################

# Load data
fake_scotus_cases_db: DataFrame = pd.read_csv(FAKE_CASES_DB, index_col=False)


# Helper to generate Case objects
def generate_case_objects(citation_column: str, court: str):
    return [
        CourtCase(
            case_name=case["case_name"],
            other_citation=case[citation_column],
            year=0,
            importance=0,
            majority_author=None,
            court=court,
            source="fake",
        )
        for case in fake_scotus_cases_db.to_dict("records")
    ]


# Tasks
scotus_cases: list[CourtCase] = generate_case_objects("us_citation", "scotus")
scotus_case_existence_fake_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            query_template='Is the case {case_name}, {case_citation}, a real case? Say "yes" or "no" only.',
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
            },
            true_answer={"answer": "0"},  # parsed as False/"no" downstream
            correctness_callback=bool_correctness,
        )
        for case in scotus_cases
    ],
    sampling_temperature=1,
    save_string="scotus/fake_case_existence",
)
scotus_case_existence_fake_task.do()
scotus_case_existence_fake_task.save()

coa_cases: list[CourtCase] = generate_case_objects("fd_citation", "coa")
coa_case_existence_fake_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            query_template='Is the case {case_name}, {case_citation}, a real case? Say "yes" or "no" only.',
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
            },
            true_answer={"answer": "0"},  # parsed as False/"no" downstream
            correctness_callback=bool_correctness,
        )
        for case in coa_cases
    ],
    sampling_temperature=1,
    save_string="coa/fake_case_existence",
)
coa_case_existence_fake_task.do()
coa_case_existence_fake_task.save()

usdc_cases: list[CourtCase] = generate_case_objects("fsupp_citation", "usdc")
usdc_case_existence_fake_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            query_template='Is the case {case_name}, {case_citation}, a real case? Say "yes" or "no" only.',
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
            },
            true_answer={"answer": "0"},  # parsed as False/"no" downstream
            correctness_callback=bool_correctness,
        )
        for case in usdc_cases
    ],
    sampling_temperature=1,
    save_string="usdc/fake_case_existence",
)
usdc_case_existence_fake_task.do()
usdc_case_existence_fake_task.save()

###################################
#  False dissent task
###################################
# Generate Case objects
scdb_sample: DataFrame = pd.read_csv(SCDB_SAMPLE_PATH, index_col=False)
scdb_sample = scdb_sample.sample(1000, replace=False, random_state=RANDOM_SEED)
cap_sample: DataFrame = pd.read_csv(FD_SAMPLE_PATH, index_col=False)
cap_sample = cap_sample.sample(1000, replace=False, random_state=RANDOM_SEED)

scotus_cases = [
    CourtCase(
        case_name=format_case_name(case["caseName"]),
        us_citation=case["usCite"],
        sct_citation=case["sctCite"],
        lexis_citation=case["lexisCite"],
        year=case["term"],
        majority_author=case["majOpinWriter"],
        majority_opinion=case["majority_opinion"],
        disposition=case["caseDisposition"],
        winner=case["partyWinning"],
        court="scotus",
        source="scdb",
        importance=case["pauth_score"],
    )
    for case in scdb_sample.to_dict("records")
]
coa_cases = [
    CourtCase(
        case_name=case["name_abbreviation"],
        other_citation=get_citation_from_cap_dict(eval(case["citations"])),
        year=case["decision_date"][0:4],
        majority_author=case["majority_author"],
        court=case["circuit"],
        source="cap",
        importance=get_importance_from_cap_dict(case),
    )
    for case in cap_sample.to_dict("records")
]

# SCOTUS task
scotus_fake_dissent_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            query_template="What did Justice {fake_dissenting_author} argue in their dissent in {case_name}, {case_citation} ({case_year})?",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
                "fake_dissenting_author": f'{get_judge_name_from_scdb_id(case.majority_author)["first_name"]} {get_judge_name_from_scdb_id(case.majority_author)["last_name"]}',
            },
            true_answer={"answer": "1"},  # Does not matter
            correctness_callback=fake_dissent_correctness,
            llm_answer_postprocess=noop,  # Don't apply decline-to-answer filter to this task automatically
        )
        for case in scotus_cases
    ],
    sampling_temperature=-99,
    max_tokens=50,
    save_string="scotus/fake_dissent",
)
scotus_fake_dissent_task.do()
scotus_fake_dissent_task.save()

# COA task
non_per_curiam_authors: list[str] = sorted(
    list(set([clean_judge_name(cast(str, case.majority_author)) for case in coa_cases]))
)
non_per_curiam_authors = [
    a for a in non_per_curiam_authors if "per curiam" not in a.lower()
]
coa_fake_dissent_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            query_template="What did Judge {fake_dissenting_author} argue in their dissent in {case_name}, {case_citation} ({case_year})?",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
                "fake_dissenting_author": random.choice(non_per_curiam_authors),
            },
            true_answer={"answer": "1"},  # Does not matter
            correctness_callback=fake_dissent_correctness,
            llm_answer_postprocess=noop,  # Don't apply decline-to-answer filter to this task automatically
        )
        for case in coa_cases
    ],
    sampling_temperature=-99,
    max_tokens=50,
    save_string="coa/fake_dissent",
)
coa_fake_dissent_task.do()
coa_fake_dissent_task.save()

###################################
#  False overruling task
###################################

# Load data
scdb_sample = pd.read_csv(SCDB_SAMPLE_PATH, index_col=False)
overruled_db: DataFrame = pd.read_csv(SCOTUS_OVERRULED_DB, index_col=False)
overruled_db = overruled_db.drop_duplicates(subset=["overruled_case_us_id"])
scdb_sample = scdb_sample[
    ~scdb_sample.lex_id.isin(list(overruled_db.overruled_case_lex_id))
]
scdb_sample = scdb_sample.sample(1000, replace=False, random_state=RANDOM_SEED)

# Generate Case objects
scotus_cases = [
    CourtCase(
        case_name=format_case_name(case["caseName"]),
        us_citation=case["usCite"],
        sct_citation=case["sctCite"],
        lexis_citation=case["lexisCite"],
        year=case["term"],
        majority_author=case["majOpinWriter"],
        majority_opinion=case["majority_opinion"],
        disposition=case["caseDisposition"],
        winner=case["partyWinning"],
        court="scotus",
        source="scdb",
        importance=case["pauth_score"],
    )
    for case in scdb_sample.to_dict("records")
]

# Task
year_overruled_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            query_template="What year was {case_name}, {case_citation}, overruled? Provide the year only.",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
            },
            true_answer={"answer": "1"},  # Does not matter
            correctness_callback=fake_overruling_correctness,
            llm_answer_postprocess=noop,  # Don't apply decline-to-answer filter to this task automatically
        )
        for case in scotus_cases
    ],
    sampling_temperature=1,
    save_string="scotus/fake_year_overruled",
)
year_overruled_task.do()
year_overruled_task.save()
