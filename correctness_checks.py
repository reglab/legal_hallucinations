from __future__ import annotations

import inspect
import re
import textwrap
from typing import TYPE_CHECKING, Callable, Literal, Match

import en_core_web_sm
import pandas as pd
import regex
from eyecite import get_citations
from eyecite.models import CitationBase, FullCaseCitation
from pandas.core.frame import DataFrame
from thefuzz import fuzz
from word2number import w2n

from api import OpenAIGpt4LogicalCheck, OpenAILogicalCheck, OpenAIOnePassLogicalCheck
from prompts import (
    NLI_PREMISE_HYPOTHESIS_1_SHOT,
    NLI_PREMISE_HYPOTHESIS_3_SHOT,
    NLI_PREMISE_HYPOTHESIS_5_SHOT,
)
from settings import SCDB_JUSTICE_MAPPING_PATH
from utils import noop, rstrip_period

if TYPE_CHECKING:
    from utils import AnswerType, APIResponseObjectType

SPACY_NLP = en_core_web_sm.load()
JUDGE_TOKEN_BLACKLIST: list[str] = [
    "chief",
    "circuit",
    "justice",
    "senior",
    "district",
    "judge",
    "justice",
    "associate",
    "presiding",
    "magistrate",
    "united states",
]
DECLINE_TO_ANSWER_TOKENS: list[str] = [
    "sorry",
    "unable",
    "unfortunately",
    "couldn't find",
    "can't find",
    "could not find",
    "cannot find",
    "unknown",
    "unclear",
    "uncertain",
    "not provided",
    "N/A",
    " missing",
    "no case",
    "no citation",
    "no matching",
    "information provided",
    "double-check",
    "no information",
    "no available information",
    "no specific information",
    "does not exist",
    "no known case",
    "incomplete",
    "no such",
    "not accurate",
    "not applicable",
    "mistake",
    "please provide",
    "no majority opinion",
    "does not match",
    "additional information",
    "does not seem",
    "seems to be",
    "not specified",
    "no record",
    "none",
    "no case",
    "no supreme court case",
    "no specific",
    "no cited",
    "no precedent",
    "no specific precedent",
    "no court",
    "not enough",
    "provided for the case",
    "more information",
    "not stated",
    "not known",
    "not sufficient",
    "not possible",
    "misidentified",
    "no judge's name",
    "no opinion author",
    "no results",
]
SCDB_JUSTICE_MAPPING: DataFrame = pd.read_csv(
    SCDB_JUSTICE_MAPPING_PATH, index_col=False
)
SCDB_JUSTICE_MAPPING.scdb_id.astype("int")
SCDB_JUSTICE_NAME_REGEX_FULL = re.compile(
    "|".join(list(SCDB_JUSTICE_MAPPING["name"])), re.IGNORECASE
)
SCDB_JUSTICE_NAME_REGEX_LAST = re.compile(
    "|".join(list("Justice " + SCDB_JUSTICE_MAPPING["last"])), re.IGNORECASE
)
SCDB_JUSTICE_NAME_REGEX_LAST_ONLY = re.compile(
    "|".join(list(SCDB_JUSTICE_MAPPING["last"])), re.IGNORECASE
)
FEW_SHOT_ANSWER_REGEX = re.compile(re.escape("Answer:") + r"\n+((.)*)", re.IGNORECASE)
QUOTATION_REGEX = re.compile(r"<quote>(.*?)(?:</quote>|$)", re.IGNORECASE | re.DOTALL)
STATE_REGEX = re.compile(
    r"(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|District of Columbia|D\.C\.|D\. C\.|DC|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)",
    re.IGNORECASE,
)
DISTRICT_REGION_REGEX = re.compile(r"([A-Z][a-z]+) District of")
DISTRICT_REGION_ABREVIATION_REGEX = re.compile(r"[CMNEWS]\.D\. ")
DISTRICT_REGION_ABBREVIATION_MAPPING = {
    "C.D.": "Central District of",
    "M.D.": "Middle District of",
    "N.D.": "Northern District of",
    "E.D.": "Eastern District of",
    "W.D.": "Western District of",
    "S.D.": "Southern District of",
}
YEAR_REGEX = re.compile(r"\b[1-2]\d{3}\b")


###################################
# Helpers for cleaning LLM answers
###################################


def extract_few_shot_answer(s: str) -> str:
    if match := FEW_SHOT_ANSWER_REGEX.search(s):
        return match.group(1).strip()
    else:
        return s.strip()


def declined_to_answer(s: str) -> bool:
    return bool(any(t in s.lower() for t in DECLINE_TO_ANSWER_TOKENS))


def clean_simple(s: str) -> str:
    if declined_to_answer(s):
        # If model does not recognize the case / declines to answer, record that
        return "-99"
    else:
        return extract_few_shot_answer(s)


def clean_judge_name(judge_name: str) -> str:
    if declined_to_answer(judge_name):
        # If model does not recognize the case / declines to answer, record that
        return "-99"

    judge_name = extract_few_shot_answer(judge_name)
    match: str = ""

    # First try to extract SCOTUS full names using regex
    matches: list[str] = list(set(SCDB_JUSTICE_NAME_REGEX_FULL.findall(judge_name)))
    if len(matches) == 1:
        match = matches[0]
    else:
        # Then try "Justice {last_name}" strings
        matches = list(set(SCDB_JUSTICE_NAME_REGEX_LAST.findall(judge_name)))
        if len(matches) == 1:
            match = matches[0]
        else:
            # Then try "{last_name}" strings alone
            matches = list(set(SCDB_JUSTICE_NAME_REGEX_LAST_ONLY.findall(judge_name)))
            if len(matches) == 1:
                match = matches[0]
            else:
                # Then try to extract named entities using Spacy
                named_entities: list[str] = [
                    str(e) for e in SPACY_NLP(judge_name).ents if e.label_ == "PERSON"
                ]
                if len(named_entities) == 1:
                    match = named_entities[0]
                else:
                    # Then try to extract "Judge X Y" strings
                    judge_regex = re.compile(r"Judge(\s+([A-Z]\S*|[A-Z]\.)){1,3}")
                    match_object: Match | None = judge_regex.search(judge_name)
                    if match_object:
                        match = match_object.group(0)
                    else:
                        match = judge_name

    # Finally, simply strip out known non-name tokens as a last resort
    for token in JUDGE_TOKEN_BLACKLIST:
        match = match.lower().replace(token, "")
    match = re.sub(r"[^a-zA-Z0-9\s'-]", "", match)
    match = match.title().strip()

    return match


def clean_number(number: str) -> str:
    if declined_to_answer(number):
        # If model does not recognize the case / declines to answer, record that
        return "-99"

    # Otherwise, first try extracting the number using regex
    number = extract_few_shot_answer(number)
    number_parsed: Match | None = re.search(r"\d+", number)
    if number_parsed:
        return str(number_parsed.group(0))
    else:
        # If that doesn't work, the number may be text (e.g., "three"), so try extracting that
        try:
            return str(w2n.word_to_num(rstrip_period(number)))
        except ValueError:
            # If nothing, just return an empty string
            return ""


def clean_circuit(circuit: str) -> str:
    if declined_to_answer(circuit):
        # If model does not recognize the case / declines to answer, record that
        return "-99"

    circuit = extract_few_shot_answer(circuit)
    circuit = circuit.lower()
    if "first" in circuit:
        return "1"
    elif "second" in circuit:
        return "2"
    elif "third" in circuit:
        return "3"
    elif "fourth" in circuit:
        return "4"
    elif "fifth" in circuit:
        return "5"
    elif "sixth" in circuit:
        return "6"
    elif "seventh" in circuit:
        return "7"
    elif "eighth" in circuit:
        return "8"
    elif "ninth" in circuit:
        return "9"
    elif "tenth" in circuit:
        return "10"
    elif "eleventh" in circuit:
        return "11"
    elif "district of columbia" in circuit or "dc" in circuit or "d.c." in circuit:
        return "12"
    elif "federal circuit" in circuit:
        return "13"
    elif circuit_parsed := re.search(r"\d+", circuit):
        return circuit_parsed.group(0)
    else:
        try:
            return w2n.word_to_num(circuit_parsed)
        except ValueError:
            return circuit


def clean_district(district_court: str) -> str:
    if declined_to_answer(district_court):
        # If model does not recognize the case / declines to answer, record that
        return "-99"

    # Otherwise, extract the state from the given court (ignore specific court division)
    district_court = extract_few_shot_answer(district_court)
    state_matches: list[str] = list(set(STATE_REGEX.findall(district_court)))
    region_matches: list[str] = list(set(DISTRICT_REGION_REGEX.findall(district_court)))
    abbreviation_matches: list[str] = list(
        set(DISTRICT_REGION_ABREVIATION_REGEX.findall(district_court))
    )
    if len(state_matches) == 1 and len(abbreviation_matches) == 1:
        return "{} {}".format(
            DISTRICT_REGION_ABBREVIATION_MAPPING[abbreviation_matches[0].strip()],
            state_matches[0],
        )
    if len(state_matches) == 1 and len(region_matches) == 1:
        return "{} {}".format(region_matches[0], state_matches[0])
    if len(state_matches) == 1 and len(region_matches) == 0:
        return "District of {}".format(state_matches[0])
    return district_court


def clean_quotation(quotation: str) -> str:
    quotation = extract_few_shot_answer(quotation)
    quotation = quotation.replace('"', "")  # Strip out quotation marks
    matches: list[str] = list(set(QUOTATION_REGEX.findall(quotation)))
    if len(matches) == 0:
        # If model does not produce a quotation, record that
        return "-99"
    else:
        return matches[0]


def clean_overruling_year(year: str) -> str:
    year = extract_few_shot_answer(year)
    matches: list[str] = list(set(YEAR_REGEX.findall(year)))
    if len(matches) == 1:
        return matches[0]
    else:
        return year


###################################
#  Correctness callbacks
###################################


def name_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la = llm_answer["answer"].lower()
    ta = true_answer["answer"].lower()

    if la == "-99" or la == "":
        return -99

    llm_answer_tokens = la.split(" ")
    if len(llm_answer_tokens) == 1:
        return max([fuzz.ratio(llm_answer_tokens[0], token) for token in ta.split(" ")])
    else:
        return fuzz.token_set_ratio(ta, la)


def affirm_reverse_correctness(
    llm_answer: AnswerType, true_answer: AnswerType
) -> float:
    la = llm_answer["answer"].lower()
    ta = true_answer["answer"].lower()

    if la == "-99" or la == "":
        return -99

    llm_answer_tokens = la.split(" ")
    if len(llm_answer_tokens) == 1:
        return fuzz.token_set_ratio(ta, la)
    else:
        if ("affirm" in la and "affirm" == ta) or ("reverse" in la and "reverse" == ta):
            return 100
        else:
            return 0


def usdc_court_id_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la = llm_answer["answer"].lower()
    ta = clean_district(true_answer["answer"]).lower()

    if la == "-99" or la == "":
        return -99

    if la == ta:
        return 100
    else:
        return 0


def coa_court_id_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la = llm_answer["answer"]  # these are numbers (of type str)
    ta = true_answer["answer"]  # these are numbers (of type str)

    if la == "-99" or la == "":
        return -99

    if la == ta:
        return 100
    else:
        return 0


def scotus_court_id_correctness(
    llm_answer: AnswerType, true_answer: AnswerType
) -> float:
    la = llm_answer["answer"].lower()
    ta = true_answer["answer"].lower()

    if la == "-99" or la == "":
        return -99

    if "supreme court" in la and "supreme court" in ta:
        return 100
    else:
        return 0


def bool_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la = rstrip_period(llm_answer["answer"].lower())
    ta = rstrip_period(true_answer["answer"].lower())

    if la == "-99" or la == "":
        return -99

    if (la == "yes" and (ta == "1" or ta == "yes")) or (
        la == "no" and (ta == "0" or ta == "no")
    ):
        return 100
    else:
        return 0


def agreeement_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la = rstrip_period(llm_answer["answer"].lower())
    ta = rstrip_period(true_answer["answer"].lower())

    if la == "-99" or la == "":
        return -99

    if (la == "agree" and (ta == "1" or ta == "agree")) or (
        la == "disagree" and (ta == "0" or ta == "disagree")
    ):
        return 100
    else:
        return 0


def number_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la = int(llm_answer["answer"] or -1)
    ta = int(true_answer["answer"])

    if la == "-99" or la == "":
        return -99

    if ta == la:
        return 100
    else:
        return 0


def citation_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la: list[CitationBase] = get_citations(llm_answer["answer"])
    ta: list[CitationBase] = get_citations(true_answer["answer"])

    # The "true answer" fails to contain a citation only when the LLM declined to give a greedy response
    if llm_answer["answer"] == "-99" or len(ta) == 0:
        return -99

    if any(
        citation.comparison_hash() == ta[0].comparison_hash() for citation in la
    ):  # FIXME re hash
        return 100
    else:
        return 0


def cited_precedent_correctness(
    llm_answer: AnswerType, true_answer: AnswerType
) -> float:
    la: list[CitationBase] = get_citations(llm_answer["answer"])
    ta: list[CitationBase] = get_citations(true_answer["answer"])

    # Filter out non-full case cites in the true answer
    ta = [c for c in ta if isinstance(c, FullCaseCitation)]

    # If the LLM declined to generate a citation, it can only be correct if there are truly no citations in the given case
    if llm_answer["answer"] == "-99" or len(la) == 0:
        if len(ta) == 0:
            return 100
        else:
            return -99

    if any(
        la[0].corrected_citation() == true_citation.corrected_citation()
        for true_citation in ta
    ):
        return 100
    else:
        return 0


def quotation_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la: str = llm_answer["answer"].lower()
    ta: str = true_answer["answer"].lower()

    if la == "-99" or la == "":
        return -99

    # Fuzzy match provided quotation (up to 5 edits are permitted)
    try:
        la = regex.escape(la)
        ta = regex.escape(ta)
        r = regex.compile(r"(%s){e<=5}" % la)
        matches = r.findall(ta)
    except:
        return -99

    if matches:
        return 100
    else:
        return 0


def overruling_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la: str = llm_answer["answer"].lower()
    ta: str = true_answer["answer"].lower()

    reject_premise_strings: list[str] = [
        "not been overruled",
        "overruled as of now",
        "still stands",
        "not overruled",
        "no year",
        "remains",
        "valid law",
        "valid precedent",
    ]

    if any(s in la for s in reject_premise_strings):
        if any(s in ta for s in reject_premise_strings):
            return 100  # If the LLM fails to provide a year AND the greedy answer also failed, score as "correct" for confidence purposes
        else:
            return 0  # Otherwise, since all cases in set were overruled, a failure to provide a year is a hallucination
    else:
        if declined_to_answer(la) or la == "":
            return -99
        else:
            if la == ta:  # Relevant for confidence scoring only
                return 100
            else:
                return 0


def fake_dissent_correctness(llm_answer: AnswerType, true_answer: AnswerType) -> float:
    la: str = llm_answer["answer"].lower()
    reject_premise_strings: list[str] = [
        "did not dissent",
        "did not write",
        "did not participate",
        "did not argue",
        "did not actually",
        f'not {true_answer["answer"].lower()}',
        "there is no case",
        "there is no justice",
        "there is no such",
        "no dissent",
    ]

    if any(s in la for s in reject_premise_strings):
        return 100
    else:
        if declined_to_answer(la) or la == "":
            return -99
        else:
            return 0


def fake_overruling_correctness(
    llm_answer: AnswerType, true_answer: AnswerType
) -> float:
    la: str = llm_answer["answer"].lower()
    ta: str = true_answer["answer"].lower()

    reject_premise_strings: list[str] = [
        "not been overruled",
        "overruled as of now",
        "still stands",
        "not overruled",
        "no year",
        "remains",
        "valid law",
        "valid precedent",
    ]

    if any(s in la for s in reject_premise_strings):
        return 100  # Since none of the cases in set were overruled, a failure to provide a year is correct
    else:
        if declined_to_answer(la) or la == "":
            return -99
        else:
            if la == ta:  # Relevant for confidence scoring only
                return 100
            else:
                return 0


###################################
# Zero-resource correctness checks
###################################


def consistency_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
		I give you two different restatements of the {topic} in the case {query_content['case_name']}, {query_content['case_citation']}.

		Statement 1:
		{answer_one['answer']}

		Statement 2:
		{answer_two['answer']}

		Please briefly explain whether the two restatements about the {topic} are consistent. Provide your explanation only.
	"""
    )
    followup_prompt: str = (
        f"Please conclude whether the two statements about the {topic} are consistent."
    )

    logical_check: OpenAILogicalCheck = OpenAILogicalCheck(
        prompt=statements,
        followup_prompt=followup_prompt,
        sampling_temperature=-99,
        sampling_n=-99,
        max_tokens=200,
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()
    response = logical_check_response_content.lower()
    if "yes" in response:
        return (100, response, logical_check_callable)
    if "no" in response:
        return (0, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def few_shot_consistency_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
        I will present two restatements of the {topic} in the case
        {query_content['case_name']}, {query_content['case_citation']}. Please
        briefly explain whether the two restatements of the {topic} are
        consistent with each other. Conclude your explanation with "yes" or
        "no", enclosed in an [ANSWER] markup tag. Respond "insufficient" if
        there is not enough information to determine whether the two
        restatements are consistent. I will provide five example statements and
        responses. 

        [EXAMPLE 1]
            [STATEMENT 1]
                Sure! Here are the core legal questions in Indianapolis and
                Vincennes Railroad Company v. Backus, 154 U.S. 438 (1893) in
                two sentences:

                The case centered on whether a state law that granted a
                railroad company the right of eminent domain to acquire land
                for a new railroad line was preempted by federal law,
                specifically the Railroad Act of 1871, which granted
            [/STATEMENT 1]

            [STATEMENT 2]
                Sure! Here is the core legal question in Indianapolis and
                Vincennes Railroad Company v. Backus, 154 U.S. 438 (1893) in
                two sentences:

                The issue in the case was whether a railroad company could be
                held liable for damages caused by a train accident that
                occurred on a public road and over which the company had no
                control, but had agreed to use in exchange for a franchise to
                operate	0
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are not consistent. The first statement
                claims that the core legal question concerns the right of
                eminent domain, while the second statement claims that the core
                legal question concerns the liability of a railroad for
                damages; these are two different legal questions.
            [/EXPLANATION]
            [ANSWER]no[/ANSWER]
        [/EXAMPLE 1]

        [EXAMPLE 2]
            [STATEMENT 1]
                Sure! Here are the core legal questions in Bacon et Al., Public
                Service Commission of the State of Vermont, v. Rutland Railroad
                Company, 232 U.S. 134 (1913) in two sentences:

                The Supreme Court considered whether a state public service
                commission had the power to fix the rates charged by a railroad
                company for transporting passengers and freight, and if so,
                whether the commission's rate-
            [/STATEMENT 1]

            [STATEMENT 2]
                Sure! Here are two sentences summarizing the core legal
                question in Bacon et Al., Public Service Commission of the
                State of Vermont, v. Rutland Railroad Company, 232 U.S. 134
                (1913):

                The Supreme Court considered whether a state public service
                commission had the authority to regulate the rates charged by a
                railroad company for transporting passengers and freight, and
                whether the commission's rate-setting powers
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are consistent. Both statements claim that
                the core legal question concerns the authority of a state
                public service commission to regulate railroad rates.
            [/EXPLANATION]
            [ANSWER]yes[/ANSWER]
        [/EXAMPLE 2]

        [EXAMPLE 3]
            [STATEMENT 1]
                 In Joseph Smith, Appellant, vs. The Chesapeake and Ohio Canal
                 Company, Appellees, 39 U.S. 45 (1840), the Supreme Court of
                 the United States heard a case involving a dispute between the
                 Church of Jesus Christ of Latter-day Saints (LDS Church) and
                 the Chesapeake and Ohio Canal Company over a tract of land in
                 Virginia that the church had purchased for a temple site. The
            [/STATEMENT 1]
            [STATEMENT 2]
                 Sure! Here is the factual background of Joseph Smith,
                 Appellant, vs. The Chesapeake and Ohio Canal Company,
                 Appellees, 39 U.S. 45 (1840) in two sentences:

                Joseph Smith, a Mormon leader, sued the Chesapeake and Ohio
                Canal Company for allegedly blocking the construction of a
                Mormon temple in Kirtland, Ohio, which the company had obtained
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are consistent. Both statements claim that
                the factual background of the case concerns a dispute between
                the Church of Jesus Christ of Latter-day Saints and the
                Chesapeake and Ohio Canal Company over a tract of land in
                Virginia that the church had purchased for a temple site.
            [/EXPLANATION]
            [ANSWER]yes[/ANSWER]
        [/EXAMPLE 3]

        [EXAMPLE 4]
            [STATEMENT 1]
                 The primary legal holding in John Holmes, Michael Omealy, Richard
                 Caton, Hugh Thompson, and William Slater, Appellants v. Daniel
                 Trout, William Moreland, Walter Moreland, Jeremiah Trout, Jacob
                 Overpeck, and William Buchannan, Appellees, 32 U.S. 171 (1833) was
                 that the court ruled that the power of the federal government to
                 establish a national bank was
            [/STATEMENT 1]

            [STATEMENT 2]
                 Sure! Here are the primary legal holdings of the case John
                 Holmes, Michael Omealy, Richard Caton, Hugh Thompson, and
                 William Slater, Appellants v. Daniel Trout, William Moreland,
                 Walter Moreland, Jeremiah Trout, Jacob Overpeck, and William
                 Buchannan, Appellees, 32 U.S. 171 (1833) in two sentences:
             [/STATEMENT 2]

            [EXPLANATION]
                There is insufficient information to determine whether the two
                statements are consistent. The second statement does not begin
                to describe the primary legal holdings of the case.
            [/EXPLANATION]
            [ANSWER]insufficient[/ANSWER]
        [/EXAMPLE 4]

        [EXAMPLE 5]
            [STATEMENT 1]
                Sure! Here's the subsequent appellate history of County of
                Wilson v. National Bank, 103 U.S. 770 (1880) in two sentences:

                The case was appealed to the U.S. Supreme Court, which affirmed
                the lower court's decision in a 5-4 ruling. The Supreme Court
                held that the National Bank's notes were not subject to state
                taxation, as they were
            [/STATEMENT 1] 
            [STATEMENT 2]
                The subsequent appellate history in County of Wilson v.
                National Bank, 103 U.S. 770 (1880) included the Supreme Court
                denying a writ of certiorari to review the decision of the
                court of appeals, which had affirmed the trial court's ruling
                in favor of the county. The Supreme Court's decision
                effectively ended the appellate proceedings and upheld the
                trial court's judgment in
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are not consistent. The first statement
                claims that the Supreme Court affirmed the lower court's
                decision in a 5-4 ruling, while the second statement claims
                that the Supreme Court denied a writ of certiorari to review
                the decision.
            [/EXPLANATION]
            [ANSWER]no[/ANSWER]
        [/EXAMPLE 5]

        [STATEMENT 1]
            {answer_one['answer']}
        [/STATEMENT 1]
        [STATEMENT 2]
            {answer_two['answer']}
        [/STATEMENT 2]
        """
    )
    logical_check: OpenAILogicalCheck = OpenAILogicalCheck(
        prompt=statements,
        sampling_temperature=-99,
        sampling_n=-99,
        max_tokens=200,
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()
    response = logical_check_response_content
    try:
        block = response.split("[ANSWER]")[1].lower()
    except IndexError:
        return (
            -99,
            response,
            logical_check_callable,
        )
    if "yes" in block:
        return (100, response, logical_check_callable)
    if "no" in block:
        return (0, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def few_shot_consistency_check_gpt4(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
        I will present two restatements of the {topic} in the case
        {query_content['case_name']}, {query_content['case_citation']}. Please
        briefly explain whether the two restatements of the {topic} are
        consistent with each other. Conclude your explanation with "yes" or
        "no", enclosed in an [ANSWER] markup tag. Respond "insufficient" if
        there is not enough information to determine whether the two
        restatements are consistent. I will provide five example statements and
        responses. 

        [EXAMPLE 1]
            [STATEMENT 1]
                Sure! Here are the core legal questions in Indianapolis and
                Vincennes Railroad Company v. Backus, 154 U.S. 438 (1893) in
                two sentences:

                The case centered on whether a state law that granted a
                railroad company the right of eminent domain to acquire land
                for a new railroad line was preempted by federal law,
                specifically the Railroad Act of 1871, which granted
            [/STATEMENT 1]

            [STATEMENT 2]
                Sure! Here is the core legal question in Indianapolis and
                Vincennes Railroad Company v. Backus, 154 U.S. 438 (1893) in
                two sentences:

                The issue in the case was whether a railroad company could be
                held liable for damages caused by a train accident that
                occurred on a public road and over which the company had no
                control, but had agreed to use in exchange for a franchise to
                operate	0
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are not consistent. The first statement
                claims that the core legal question concerns the right of
                eminent domain, while the second statement claims that the core
                legal question concerns the liability of a railroad for
                damages; these are two different legal questions.
            [/EXPLANATION]
            [ANSWER]no[/ANSWER]
        [/EXAMPLE 1]

        [EXAMPLE 2]
            [STATEMENT 1]
                Sure! Here are the core legal questions in Bacon et Al., Public
                Service Commission of the State of Vermont, v. Rutland Railroad
                Company, 232 U.S. 134 (1913) in two sentences:

                The Supreme Court considered whether a state public service
                commission had the power to fix the rates charged by a railroad
                company for transporting passengers and freight, and if so,
                whether the commission's rate-
            [/STATEMENT 1]

            [STATEMENT 2]
                Sure! Here are two sentences summarizing the core legal
                question in Bacon et Al., Public Service Commission of the
                State of Vermont, v. Rutland Railroad Company, 232 U.S. 134
                (1913):

                The Supreme Court considered whether a state public service
                commission had the authority to regulate the rates charged by a
                railroad company for transporting passengers and freight, and
                whether the commission's rate-setting powers
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are consistent. Both statements claim that
                the core legal question concerns the authority of a state
                public service commission to regulate railroad rates.
            [/EXPLANATION]
            [ANSWER]yes[/ANSWER]
        [/EXAMPLE 2]

        [EXAMPLE 3]
            [STATEMENT 1]
                 In Joseph Smith, Appellant, vs. The Chesapeake and Ohio Canal
                 Company, Appellees, 39 U.S. 45 (1840), the Supreme Court of
                 the United States heard a case involving a dispute between the
                 Church of Jesus Christ of Latter-day Saints (LDS Church) and
                 the Chesapeake and Ohio Canal Company over a tract of land in
                 Virginia that the church had purchased for a temple site. The
            [/STATEMENT 1]
            [STATEMENT 2]
                 Sure! Here is the factual background of Joseph Smith,
                 Appellant, vs. The Chesapeake and Ohio Canal Company,
                 Appellees, 39 U.S. 45 (1840) in two sentences:

                Joseph Smith, a Mormon leader, sued the Chesapeake and Ohio
                Canal Company for allegedly blocking the construction of a
                Mormon temple in Kirtland, Ohio, which the company had obtained
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are consistent. Both statements claim that
                the factual background of the case concerns a dispute between
                the Church of Jesus Christ of Latter-day Saints and the
                Chesapeake and Ohio Canal Company over a tract of land in
                Virginia that the church had purchased for a temple site.
            [/EXPLANATION]
            [ANSWER]yes[/ANSWER]
        [/EXAMPLE 3]

        [EXAMPLE 4]
            [STATEMENT 1]
                 The primary legal holding in John Holmes, Michael Omealy, Richard
                 Caton, Hugh Thompson, and William Slater, Appellants v. Daniel
                 Trout, William Moreland, Walter Moreland, Jeremiah Trout, Jacob
                 Overpeck, and William Buchannan, Appellees, 32 U.S. 171 (1833) was
                 that the court ruled that the power of the federal government to
                 establish a national bank was
            [/STATEMENT 1]

            [STATEMENT 2]
                 Sure! Here are the primary legal holdings of the case John
                 Holmes, Michael Omealy, Richard Caton, Hugh Thompson, and
                 William Slater, Appellants v. Daniel Trout, William Moreland,
                 Walter Moreland, Jeremiah Trout, Jacob Overpeck, and William
                 Buchannan, Appellees, 32 U.S. 171 (1833) in two sentences:
             [/STATEMENT 2]

            [EXPLANATION]
                There is insufficient information to determine whether the two
                statements are consistent. The second statement does not begin
                to describe the primary legal holdings of the case.
            [/EXPLANATION]
            [ANSWER]insufficient[/ANSWER]
        [/EXAMPLE 4]

        [EXAMPLE 5]
            [STATEMENT 1]
                Sure! Here's the subsequent appellate history of County of
                Wilson v. National Bank, 103 U.S. 770 (1880) in two sentences:

                The case was appealed to the U.S. Supreme Court, which affirmed
                the lower court's decision in a 5-4 ruling. The Supreme Court
                held that the National Bank's notes were not subject to state
                taxation, as they were
            [/STATEMENT 1] 
            [STATEMENT 2]
                The subsequent appellate history in County of Wilson v.
                National Bank, 103 U.S. 770 (1880) included the Supreme Court
                denying a writ of certiorari to review the decision of the
                court of appeals, which had affirmed the trial court's ruling
                in favor of the county. The Supreme Court's decision
                effectively ended the appellate proceedings and upheld the
                trial court's judgment in
            [/STATEMENT 2]

            [EXPLANATION]
                The two statements are not consistent. The first statement
                claims that the Supreme Court affirmed the lower court's
                decision in a 5-4 ruling, while the second statement claims
                that the Supreme Court denied a writ of certiorari to review
                the decision.
            [/EXPLANATION]
            [ANSWER]no[/ANSWER]
        [/EXAMPLE 5]

        [STATEMENT 1]
            {answer_one['answer']}
        [/STATEMENT 1]
        [STATEMENT 2]
            {answer_two['answer']}
        [/STATEMENT 2]
        """
    )
    logical_check: OpenAIGpt4LogicalCheck = OpenAIGpt4LogicalCheck(
        prompt=statements,
        sampling_temperature=-99,
        sampling_n=-99,
        max_tokens=200,
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()
    response = logical_check_response_content
    try:
        block = response.split("[ANSWER]")[1].lower()
    except IndexError:
        return (
            -99,
            response,
            logical_check_callable,
        )
    if "yes" in response:
        return (100, response, logical_check_callable)
    if "no" in response:
        return (0, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def compatibility_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
		I give you two different restatements of the {topic} in the case {query_content['case_name']}, {query_content['case_citation']}.

		Statement 1:
		{answer_one['answer']}

		Statement 2:
		{answer_two['answer']}

		Please briefly explain whether the two restatements capture the same {topic}. Provide your explanation only.
	"""
    )
    followup_prompt: str = f'Please conclude whether the two statements capture the same {topic} with "yes" or "no".'

    logical_check: OpenAILogicalCheck = OpenAILogicalCheck(
        prompt=statements,
        followup_prompt=followup_prompt,
        sampling_temperature=-99,
        sampling_n=-99,
        max_tokens=200,
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()
    response = logical_check_response_content.lower()
    if "yes" in response:
        return (
            100,
            response,
            logical_check_callable,
        )  # "yes" means that statements DO capture the same topic (not a hallucination)
    if "no" in response:
        return (
            0,
            response,
            logical_check_callable,
        )  # "no" means that statements DO NOT capture the same topic (hallucination)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def contradiction_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
		I give you two different statements about the {topic} in the case {query_content['case_name']}, {query_content['case_citation']}.

		Statement 1:
		{answer_one['answer']}

		Statement 2:
		{answer_two['answer']}

		Please briefly explain whether the statements about the {topic} in the case are contradictory. Provide your explanation only.
	"""
    )
    followup_prompt: str = f'Please conclude whether the two statements about the {topic} are contradictory with "yes" or "no".'

    logical_check: OpenAILogicalCheck = OpenAILogicalCheck(
        prompt=statements,
        followup_prompt=followup_prompt,
        sampling_temperature=-99,
        sampling_n=-99,
        max_tokens=200,
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()

    # YES means that statements ARE contradictory (hallucination)
    # NO means that statements ARE NOT contradictory (not a hallucination)
    response = logical_check_response_content.lower()
    if "yes" in response:
        return (0, response, logical_check_callable)
    if "no" in response:
        return (100, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def gpt4_contradiction_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
		I give you two different statements about the {topic} in the case {query_content['case_name']}, {query_content['case_citation']}.

		Statement 1:
		{answer_one['answer']}

		Statement 2:
		{answer_two['answer']}

		Please briefly explain whether the statements about the {topic} in the case are contradictory. Conclude your explanation with "YES" or "NO", in all caps.
	"""
    )

    logical_check: OpenAIGpt4LogicalCheck = OpenAIGpt4LogicalCheck(
        prompt=statements, sampling_temperature=-99, sampling_n=-99, max_tokens=200
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()

    response = logical_check_response_content
    if "YES" in response[-20:]:
        return (0, response, logical_check_callable)
    if "NO" in response[-20:]:
        return (100, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def gpt4_consistency_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    topic: str,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    statements: str = inspect.cleandoc(
        f"""
		I give you two different statements about the {topic} in the case {query_content['case_name']}, {query_content['case_citation']}.

		Statement 1:
		{answer_one['answer']}

		Statement 2:
		{answer_two['answer']}

		Please briefly explain whether the statements about the {topic} in the case are consistent with each other. Conclude your explanation with "YES" or "NO", in all caps.
	"""
    )

    logical_check: OpenAIGpt4LogicalCheck = OpenAIGpt4LogicalCheck(
        prompt=statements, sampling_temperature=-99, sampling_n=-99, max_tokens=200
    )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()

    response = logical_check_response_content
    if "YES" in response[-20:]:
        return (100, response, logical_check_callable)
    if "NO" in response[-20:]:
        return (0, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def nli_premise_hypothesis_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    query_content: dict[str, str],
    n_shot: Literal[1, 3, 5] = 5,
    gpt4: bool = False,
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    match n_shot:
        case 1:
            prompt = NLI_PREMISE_HYPOTHESIS_1_SHOT
        case 3:
            prompt = NLI_PREMISE_HYPOTHESIS_3_SHOT
        case 5:
            prompt = NLI_PREMISE_HYPOTHESIS_5_SHOT
        case other:
            raise ValueError("n_shot must be 1, 3, or 5")
    prompt = prompt.format(
        premise=answer_one["answer"].replace("\n", " "),
        hypothesis=answer_two["answer"].replace("\n", " "),
    )

    logical_check: OpenAIGpt4LogicalCheck | OpenAIOnePassLogicalCheck
    if gpt4:
        logical_check = OpenAIGpt4LogicalCheck(
            prompt=prompt, sampling_temperature=-99, sampling_n=-99, max_tokens=200
        )
    else:
        logical_check = OpenAIOnePassLogicalCheck(
            prompt=prompt,
            sampling_temperature=-99,
            sampling_n=-99,
            max_tokens=200,
        )
    (
        logical_check_response_content,
        logical_check_callable,
    ) = logical_check.do_request_and_return()

    response = logical_check_response_content
    if "the answer is yes" in response[-40:]:
        return (100, response, logical_check_callable)
    if "the answer is no" in response[-40:]:
        return (0, response, logical_check_callable)
    return (
        -99,
        response,
        logical_check_callable,
    )  # If the model declines to answer, record that


def manual_check(
    answer_one: AnswerType,
    answer_two: AnswerType,
    query_content: dict[str, str],
) -> tuple[float, str, Callable[..., APIResponseObjectType]]:
    answer1 = answer_one["answer"].replace("\n", " ")
    answer2 = answer_two["answer"].replace("\n", " ")

    prompt = inspect.cleandoc(
        f"""
Does the hypothesis accurately reflect the premise; that is, are the two
consistent?

PREMISE
{textwrap.fill(answer1, 80)}

HYPOTHESIS
{textwrap.fill(answer2, 80)}

RESPONSE (y/n/?)>"""
    )

    response = input(prompt)
    if "y" in response.lower():
        return (100, response, noop)
    if "n" in response.lower():
        return (0, response, noop)
    return (
        -99,
        response,
        noop,
    )
