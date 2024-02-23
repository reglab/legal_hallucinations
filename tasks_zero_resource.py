import argparse
import random
from functools import partial

import pandas as pd
from pandas.core.frame import DataFrame

from api import (
    GooglePaLMCompletion,
    LlamaChat,
    OpenAIChat,
    OpenAIChatGpt4,
    TogetherAiLlamaChat,
)
from correctness_checks import nli_premise_hypothesis_check
from models import CourtCase, Query, Task
from settings import FD_SAMPLE_PATH, FSUPP_SAMPLE_PATH, RANDOM_SEED, SCDB_SAMPLE_PATH
from utils import (
    APIBackendType,
    format_case_name,
    get_citation_from_cap_dict,
    get_importance_from_cap_dict,
    get_majority_opinion_from_cap_dict,
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

# Load data
scdb_sample: DataFrame = pd.read_csv(SCDB_SAMPLE_PATH, index_col=False)
coa_sample: DataFrame = pd.read_csv(FD_SAMPLE_PATH, index_col=False)
usdc_sample: DataFrame = pd.read_csv(FSUPP_SAMPLE_PATH, index_col=False)

# Generate Case objects
scotus_cases: list[CourtCase] = [
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
random.seed(RANDOM_SEED)
scotus_cases = random.sample(scotus_cases, 100)

coa_cases: list[CourtCase] = [
    CourtCase(
        case_name=case["name_abbreviation"],
        other_citation=get_citation_from_cap_dict(eval(case["citations"])),
        year=case["decision_date"][0:4],
        majority_author=case["majority_author"],
        majority_opinion=get_majority_opinion_from_cap_dict(case),
        court=case["circuit"],
        source="cap",
        importance=get_importance_from_cap_dict(case),
    )
    for case in coa_sample.to_dict("records")
]
random.seed(RANDOM_SEED)
coa_cases = random.sample(coa_cases, 100)


usdc_cases: list[CourtCase] = [
    CourtCase(
        case_name=case["name_abbreviation"],
        other_citation=get_citation_from_cap_dict(eval(case["citations"])),
        year=case["decision_date"][0:4],
        majority_author=case["majority_author"],
        majority_opinion=get_majority_opinion_from_cap_dict(case),
        court=eval(case["court"])["slug"],
        state=case["state"],
        source="cap",
        importance=get_importance_from_cap_dict(case),
    )
    for case in usdc_sample.to_dict("records")
]
random.seed(RANDOM_SEED)
usdc_cases = random.sample(usdc_cases, 100)


def get_case_citation(case: CourtCase) -> str:
    if case.court == "scotus":
        if case.us_citation and not pd.isna(case.us_citation):
            return case.us_citation
        elif case.sct_citation and not pd.isna(case.sct_citation):
            return case.sct_citation
        else:
            return ""
    else:
        if case.other_citation and not pd.isna(case.other_citation):
            return case.other_citation
        else:
            return ""


for court, cases in [
    ("scotus", scotus_cases),
    ("coa", coa_cases),
    ("usdc", usdc_cases),
]:
    posture_task: Task = Task(
        api_backend_type=CURRENT_API,
        queries=[
            Query(
                test_case=case,
                system_message="No more than two sentences.",
                query_template="What was the procedural posture in {case_name}, {case_citation} ({case_year})? {system_message}",
                query_content={
                    "case_name": format_case_name(case.case_name),
                    "case_citation": get_case_citation(case),
                    "case_year": str(case.year),
                },
                true_answer=None,
                correctness_callback=partial(
                    nli_premise_hypothesis_check,
                    n_shot=3,
                    gpt4=True,
                ),  # type: ignore
            )
            for case in cases
        ],
        sampling_temperature=0.999,
        save_string=f"{court}/posture",
        max_tokens=256,
    )

    core_legal_question_task: Task = Task(
        api_backend_type=CURRENT_API,
        queries=[
            Query(
                test_case=case,
                system_message="No more than two sentences.",
                query_template="What was the core legal question in {case_name}, {case_citation} ({case_year})? {system_message}",
                query_content={
                    "case_name": format_case_name(case.case_name),
                    "case_citation": get_case_citation(case),
                    "case_year": str(case.year),
                },
                true_answer=None,
                correctness_callback=partial(
                    nli_premise_hypothesis_check,
                    n_shot=3,
                    gpt4=True,  # type: ignore
                ),
            )
            for case in cases
        ],
        sampling_temperature=0.999,
        save_string=f"{court}/core_legal_question",
        max_tokens=256,
    )

    holding_task: Task = Task(
        api_backend_type=CURRENT_API,
        queries=[
            Query(
                test_case=case,
                system_message="No more than two sentences.",
                query_template="What was the primary legal holding in {case_name}, {case_citation} ({case_year})? {system_message}",
                query_content={
                    "case_name": format_case_name(case.case_name),
                    "case_citation": get_case_citation(case),
                    "case_year": str(case.year),
                },
                true_answer=None,
                correctness_callback=partial(
                    nli_premise_hypothesis_check,
                    n_shot=3,
                    gpt4=True,  # type: ignore
                ),
            )
            for case in cases
        ],
        sampling_temperature=0.999,
        save_string=f"{court}/holding",
        max_tokens=256,
    )

    factual_background_task: Task = Task(
        api_backend_type=CURRENT_API,
        queries=[
            Query(
                test_case=case,
                system_message="No more than two sentences.",
                query_template="What was the factual background in {case_name}, {case_citation} ({case_year})? {system_message}",
                query_content={
                    "case_name": format_case_name(case.case_name),
                    "case_citation": get_case_citation(case),
                    "case_year": str(case.year),
                },
                true_answer=None,
                correctness_callback=partial(
                    nli_premise_hypothesis_check,
                    n_shot=3,
                    gpt4=True,  # type: ignore
                ),
            )
            for case in cases
        ],
        sampling_temperature=0.999,
        save_string=f"{court}/factual_background",
        max_tokens=256,
    )

    subsequent_history_task: Task = Task(
        api_backend_type=CURRENT_API,
        queries=[
            Query(
                test_case=case,
                system_message="No more than two sentences.",
                query_template="What was the subsequent appellate history in {case_name}, {case_citation} ({case_year})? {system_message}",
                query_content={
                    "case_name": format_case_name(case.case_name),
                    "case_citation": get_case_citation(case),
                    "case_year": str(case.year),
                },
                true_answer=None,
                correctness_callback=partial(
                    nli_premise_hypothesis_check,
                    n_shot=3,
                    gpt4=True,  # type: ignore
                ),
            )
            for case in cases
        ],
        sampling_temperature=0.999,
        save_string=f"{court}/subsequent_history",
        max_tokens=256,
    )

    for task in [
        posture_task,
        core_legal_question_task,
        holding_task,
        factual_background_task,
        subsequent_history_task,
    ]:
        task.do()
        task.save()
