from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING, Any, Type, Union

import en_core_web_sm
import pandas as pd
from google.generativeai.types.discuss_types import ChatResponse
from openai.openai_object import OpenAIObject
from pandas.core.frame import DataFrame
from titlecase import titlecase

from mappings import CAP_CIRCUIT_MAPPING, STATE_SLUG_MAPPING
from settings import SCDB_JUSTICE_MAPPING_PATH

SPACY_NLP = en_core_web_sm.load()

###################################
# Types
###################################
if TYPE_CHECKING:
    from api import APIBackend
    from models import CourtCase

    AnswerType = dict[str, str]
    APIBackendType = Type[APIBackend]
    APIResponseObjectType = Union[OpenAIObject, ChatResponse]

###################################
# General utility functions
###################################


def print_progress(iteration: int, total_iterations: int, prefix: str = "") -> None:
    iteration = iteration + 1
    percent: str = ("{0:.1f}").format(100 * (iteration / float(total_iterations)))
    filled: int = int(50 * iteration // total_iterations)
    bar: str = "█" * filled + "-" * (50 - filled)
    end = "\n" if iteration == total_iterations else " "
    print(f"\r{prefix} Progress: |{bar}| {percent}% complete.", end=end, flush=True)


def rstrip_period(s: str) -> str:
    return s.rstrip(".")


def titlecase_callback(word, **kwargs) -> str | None:
    if word in ["v.", "vs.", "et", "al."]:
        return word.lower()
    else:
        return None


def format_case_name(case_name: str) -> str:
    case_name = titlecase(case_name.lower(), callback=titlecase_callback)
    case_name = re.sub(r"\.\sV\.\s", ". v. ", case_name)
    case_name = re.sub(r"\.\sVs\.\s", ". vs. ", case_name)
    return case_name


def get_greedy_log_probs(greedy_response_object: OpenAIObject) -> float:
    return statistics.fmean(
        greedy_response_object.choices[0]["logprobs"]["token_logprobs"]
    )


def agreement_cutoff(score: str) -> int:
    return 0 if int(float(score)) <= 70 else 100


def noop(x: Any) -> Any:
    return x


###################################
# Helpers to prepare SCOTUS, COA, USDC data
###################################

SCDB_JUSTICE_MAPPING: DataFrame = pd.read_csv(
    SCDB_JUSTICE_MAPPING_PATH, index_col=False
)
SCDB_JUSTICE_MAPPING.scdb_id.astype("int")


def get_judge_name_from_scdb_id(scdb_id: int | str | None) -> dict[str, str]:
    if type(scdb_id) is int:
        lookup: DataFrame = SCDB_JUSTICE_MAPPING.loc[
            SCDB_JUSTICE_MAPPING["scdb_id"] == scdb_id
        ]

        if len(lookup) == 1:
            return {
                "first_name": lookup.iloc[0]["first"],
                "last_name": lookup.iloc[0]["last"],
            }
        else:
            raise Warning(f"Justice ID not in SCDB_JUSTICE_MAPPING: {scdb_id}")
    else:
        raise TypeError


def get_disposition_from_scdb_id(id: int | None) -> str:
    if id in [2]:
        return "affirm"
    elif id in [3, 4, 5, 8]:
        return "reverse"
    else:
        return ""


def get_disposition_from_songer_id(id: int | None) -> int | None:
    if id in [1]:
        return 1
    elif id in [2, 3, 4, 7]:
        return 0
    else:
        return None


def get_case_citation_for_scotus_case(case: CourtCase) -> str:
    if case.us_citation and not pd.isna(case.us_citation):
        return case.us_citation
    elif case.sct_citation and not pd.isna(case.sct_citation):
        return case.sct_citation
    else:
        raise KeyError


def get_circuit_from_cap_id(id: int) -> int:
    try:
        return CAP_CIRCUIT_MAPPING[id]
    except KeyError as e:
        return 99


def get_state_from_cap_slug(slug: str) -> str:
    try:
        return STATE_SLUG_MAPPING[slug]
    except KeyError:
        return STATE_SLUG_MAPPING["misc"]
    except:
        raise


def get_majority_author_from_cap_dict(case: dict) -> str | None:
    for o in case["data"]["opinions"]:
        if o["type"] == "majority" and o["author"]:
            return o["author"]
        elif o["type"] == "rehearing" and o["author"]:
            return o["author"]

    return None


def get_majority_opinion_from_cap_dict(case: dict) -> str | None:
    for o in eval(case["casebody"])["data"]["opinions"]:
        if o["type"] == "majority":
            return o["text"]
        elif o["type"] == "rehearing":
            return o["text"]

    return None


def get_citation_from_cap_dict(citations: dict) -> str:
    for c in citations:
        if c["type"] == "official":
            return c["cite"]
    raise KeyError


def get_importance_from_cap_dict(case: dict) -> float:
    try:
        return eval(case["analysis"])["pagerank"]["percentile"]
    except KeyError as e:
        Warning(f'No PageRank key for {case["id"]}')
        return 0.5
