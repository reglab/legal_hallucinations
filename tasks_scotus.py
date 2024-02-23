import argparse
import inspect
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
    affirm_reverse_correctness,
    agreeement_correctness,
    bool_correctness,
    citation_correctness,
    cited_precedent_correctness,
    clean_judge_name,
    clean_overruling_year,
    clean_quotation,
    name_correctness,
    overruling_correctness,
    quotation_correctness,
    scotus_court_id_correctness,
)
from models import CourtCase, CourtCasePair, Query, Task
from settings import SCDB_SAMPLE_PATH, SCOTUS_OVERRULED_DB, SCOTUS_SHEPARDS_SAMPLE
from utils import (
    APIBackendType,
    format_case_name,
    get_case_citation_for_scotus_case,
    get_disposition_from_scdb_id,
    get_judge_name_from_scdb_id,
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

# Generate Case objects
cases: list[CourtCase] = [
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

###################################
#  Case existence task
###################################
case_existence_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Say "yes" or "no" only.',
            query_template="Is the case {case_name}, {case_citation} ({case_year}), a real case? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": "1"},  # parsed as True/"yes" downstream
            correctness_callback=bool_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/case_existence",
)
case_existence_task.do()
case_existence_task.save()

###################################
#  Case existence task (few shot)
###################################
case_existence_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Say "yes" or "no" only.',
            query_template=inspect.cleandoc(
                """
				Is the given case a real case? {system_message}

				Examples:
				```
				Case: Brown v. Board of Education, 347 U.S. 483 (1954)
				Answer: Yes

				Case: Bowers v. Hardwick, 478 U.S. 186 (1986)
				Answer: Yes

				Case: Columbia University v. Rodham, 564 U.S. 911 (2010)
				Answer: No
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": "1"},  # parsed as True/"yes" downstream
            correctness_callback=bool_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/case_existence_few_shot",
)
case_existence_task_few_shot.do()
case_existence_task_few_shot.save()

###################################
#  Citation retrieval task
###################################
citation_retrieval_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Provide ONLY the citation in "<volume>, <reporter>, <page>" format, nothing else.',
            query_template="What is the citation for the case {case_name}? {system_message}",
            query_content={"case_name": format_case_name(case.case_name)},
            true_answer={"answer": get_case_citation_for_scotus_case(case)},
            correctness_callback=citation_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    max_tokens=30,
    save_string="scotus/citation_retrieval",
)
citation_retrieval_task.do()
citation_retrieval_task.save()

###################################
#  Citation retrieval task (few shot)
###################################
citation_retrieval_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Provide ONLY the citation in "<volume>, <reporter>, <page>" format, nothing else.',
            query_template=inspect.cleandoc(
                """
				What is the citation for the given case? {system_message}

				Examples:
				```
				Case: Brown v. Board of Education
				Answer: 347 U.S. 483

				Case: Bowers v. Hardwick
				Answer: 478 U.S. 186

				Case: McCulloch v. Maryland
				Answer: 17 U.S. 316
				```

				Case: {case_name}
				Answer:
			"""
            ),
            query_content={"case_name": format_case_name(case.case_name)},
            true_answer={"answer": get_case_citation_for_scotus_case(case)},
            correctness_callback=citation_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    max_tokens=30,
    save_string="scotus/citation_retrieval_few_shot",
)
citation_retrieval_task_few_shot.do()
citation_retrieval_task_few_shot.save()

###################################
#  Majority opinion author task
###################################
majority_author_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the first and the last name of the judge ONLY.",
            query_template="Who wrote the majority opinion in {case_name}, {case_citation} ({case_year})? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={
                "answer": f'{get_judge_name_from_scdb_id(case.majority_author)["first_name"]} {get_judge_name_from_scdb_id(case.majority_author)["last_name"]}'
            },
            correctness_callback=name_correctness,
            llm_answer_postprocess=clean_judge_name,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/majority_author",
)
majority_author_task.do()
majority_author_task.save()

###################################
#  Majority opinion author task (few shot)
###################################
majority_author_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the first and the last name of the judge ONLY.",
            query_template=inspect.cleandoc(
                """
				Who wrote the majority opinion in the given case? {system_message}

				Examples:
				```
				Case: Brown v. Board of Education, 347 U.S. 483 (1954)
				Answer: Earl Warren

				Case: Bowers v. Hardwick, 478 U.S. 186 (1986)
				Answer: Byron White

				Case: McCulloch v. Maryland, 17 U.S. 316 (1819)
				Answer: John Marshall
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={
                "answer": f'{get_judge_name_from_scdb_id(case.majority_author)["first_name"]} {get_judge_name_from_scdb_id(case.majority_author)["last_name"]}'
            },
            correctness_callback=name_correctness,
            llm_answer_postprocess=clean_judge_name,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/majority_author_few_shot",
)
majority_author_task_few_shot.do()
majority_author_task_few_shot.save()

###################################
# Affirm/reverse task
###################################
task_affirm_reverse: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Say "affirm" or "reverse" only.',
            query_template="Did the court in {case_name}, {case_citation} ({case_year}) affirm or reverse the lower court's decision? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": get_disposition_from_scdb_id(case.disposition)},
            correctness_callback=affirm_reverse_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/affirm_reverse",
)
task_affirm_reverse.do()
task_affirm_reverse.save()

###################################
# Affirm/reverse task (few shot)
###################################
task_affirm_reverse_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Say "affirm" or "reverse" only.',
            query_template=inspect.cleandoc(
                """
				Did the court in the given case affirm or reverse the lower court's decision? {system_message}

				Examples:
				```
				Case: Plessy v. Ferguson, 163 U.S. 537 (1896)
				Answer: Affirm

				Case: Bowers v. Hardwick, 478 U.S. 186 (1986)
				Answer: Reverse

				Case: McCulloch v. Maryland, 17 U.S. 316 (1819)
				Answer: Reverse
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": get_disposition_from_scdb_id(case.disposition)},
            correctness_callback=affirm_reverse_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/affirm_reverse_few_shot",
)
task_affirm_reverse_few_shot.do()
task_affirm_reverse_few_shot.save()

###################################
#  Court ID task
###################################
court_id_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the name of the court ONLY, nothing else.",
            query_template="Which court decided the case {case_name}, {case_citation} ({case_year})? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": "Supreme Court"},
            correctness_callback=scotus_court_id_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/court_id",
)
court_id_task.do()
court_id_task.save()

###################################
#  Court ID task (few shot)
###################################
court_id_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the name of the court ONLY, nothing else.",
            query_template=inspect.cleandoc(
                """
				Which court decided the given case? {system_message}

				Examples:
				```
				Case: Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2012)
				Answer: Second Circuit

				Case: Durham v. United States, 214 F.2d 862 (1954)
				Answer: D.C. Circuit

				Case: Bowers v. Hardwick (1986)
				Answer: Supreme Court
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": "Supreme Court"},
            correctness_callback=scotus_court_id_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="scotus/court_id_few_shot",
)
court_id_task_few_shot.do()
court_id_task_few_shot.save()

###################################
# Quotation task
###################################
quotation_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="The quotation MUST be word-for-word from the majority opinion. Wrap the quotation in <quote></quote> tags.",
            query_template="Provide a verbatim quotation from the majority opinion in the case {case_name}, {case_citation} ({case_year}). {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=quotation_correctness,
            llm_answer_postprocess=clean_quotation,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="scotus/quotation",
)
quotation_task.do()
quotation_task.save()

###################################
# Quotation task (few shot)
###################################
quotation_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="The quotation MUST be word-for-word from the majority opinion. Wrap the quotation in <quote></quote> tags.",
            query_template=inspect.cleandoc(
                """
				Provide a verbatim quotation from the majority opinion in the given case. {system_message}

				Examples:
				```
				Case: Brown v. Board of Education, 347 U.S. 483 (1954)
				Answer: <quote>We conclude that in the field of public education the doctrine of "separate but equal" has no place.</quote>

				Case: Bowers v. Hardwick, 478 U.S. 186 (1986)
				Answer: <quote>It is obvious to us that neither of these formulations would extend a fundamental right to homosexuals to engage in acts of consensual sodomy.</quote>

				Case: McConnell v. Federal Election Commission, 540 U.S. 93 (2003)
				Answer: <quote>Our cases have made clear that the prevention of corruption or its appearance constitutes a sufficiently important interest to justify political contribution limits.</quote>
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=quotation_correctness,
            llm_answer_postprocess=clean_quotation,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="scotus/quotation_few_shot",
)
quotation_task_few_shot.do()
quotation_task_few_shot.save()

###################################
# Cited precedent task
###################################
cited_precedent_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Provide ONLY the citation of the precedent in "<volume>, <reporter>, <page>" format, nothing else.',
            query_template="What is a precedent that is cited in the majority opinion of the case {case_name}, {case_citation} ({case_year})? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=cited_precedent_correctness,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="scotus/cited_precedent",
)
cited_precedent_task.do()
cited_precedent_task.save()

###################################
# Cited precedent task (few shot)
###################################
cited_precedent_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Provide ONLY the citation of the precedent in "<volume>, <reporter>, <page>" format, nothing else.',
            query_template=inspect.cleandoc(
                """
				What is a precedent that is cited in the majority opinion of the given case? {system_message}

				Examples:
				```
				Case: Brown v. Board of Education, 347 U.S. 483 (1954)
				Answer: Plessy v. Ferguson, 163 U.S. 537

				Case: Bowers v. Hardwick, 478 U.S. 186 (1986)
				Answer: Griswold v. Connecticut, 381 U.S. 479

				Case: McConnell v. Federal Election Commission, 540 U.S. 93 (2003)
				Answer: Buckley v. Valeo, 424 U.S. 1
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=cited_precedent_correctness,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="scotus/cited_precedent_few_shot",
)
cited_precedent_task_few_shot.do()
cited_precedent_task_few_shot.save()

###################################
# Doctrinal agreement task
###################################

# Load data
scotus_shepards_sample: DataFrame = pd.read_csv(SCOTUS_SHEPARDS_SAMPLE, index_col=False)

# Generate CasePair objects
case_pairs: list[CourtCasePair] = [
    CourtCasePair(
        citing_case=CourtCase(
            case_name=format_case_name(case_pair["citing_case_name"]),
            us_citation=case_pair["citing_case_us_cite"],
            year=case_pair["citing_case_year"],
            importance=0,
            majority_author=None,
            court="scotus",
            source="shepards",
        ),
        cited_case=CourtCase(
            case_name=format_case_name(case_pair["cited_case_name"]),
            us_citation=case_pair["cited_case_us_cite"],
            year=case_pair["cited_case_year"],
            importance=0,
            majority_author=None,
            court="scotus",
            source="shepards",
        ),
        positive_relationship=bool(case_pair["agree"]),
        source="shepards",
    )
    for case_pair in scotus_shepards_sample.to_dict("records")
]

# Task
doctrinal_agreement_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case_pair,
            system_message='Say "agree" or "disagree" only.',
            query_template='Do the cases "{citing_case_name}, {citing_case_citation} ({citing_case_year})" and "{cited_case_name}, {cited_case_citation} ({cited_case_year})" agree or disagree with each other? {system_message}',
            query_content={
                "citing_case_name": format_case_name(case_pair.citing_case.case_name),
                "citing_case_citation": get_case_citation_for_scotus_case(
                    case_pair.citing_case
                ),
                "citing_case_year": str(case_pair.citing_case.year),
                "cited_case_name": format_case_name(case_pair.cited_case.case_name),
                "cited_case_citation": get_case_citation_for_scotus_case(
                    case_pair.cited_case
                ),
                "cited_case_year": str(case_pair.cited_case.year),
            },
            true_answer={
                "answer": str(int(case_pair.positive_relationship))
            },  # parsed as True/"yes" downstream
            correctness_callback=agreeement_correctness,
        )
        for case_pair in case_pairs
    ],
    sampling_temperature=1,
    save_string="scotus/doctrinal_agreement",
)
doctrinal_agreement_task.do()
doctrinal_agreement_task.save()

# Task (few shot)
doctrinal_agreement_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case_pair,
            system_message='Say "agree" or "disagree" only.',
            query_template=inspect.cleandoc(
                """
				Do the two given cases agree or disagree with each other? {system_message}

				Examples:
				```
				Case 1: Brown v. Board of Education, 347 U.S. 483 (1954)
				Case 2: Plessy v. Ferguson, 163 U.S. 537 (1896)
				Answer: Disagree

				Case 1: Youngstown Sheet & Tube Co. v. Sawyer, 343 U.S. 579 (1952)
				Case 2: Medellin v. Texas, 552 U.S. 491 (2008)
				Answer: Agree

				Case 1: Whitney v. California, 274 U.S. 357 (1927)
				Case 2: Brandenburg v. Ohio, 395 U.S. 444 (1969)
				Answer: Disagree
				```

				Case 1: {citing_case_name}, {citing_case_citation} ({citing_case_year})
				Case 2: {cited_case_name}, {cited_case_citation} ({cited_case_year})
				Answer:
			"""
            ),
            query_content={
                "citing_case_name": format_case_name(case_pair.citing_case.case_name),
                "citing_case_citation": get_case_citation_for_scotus_case(
                    case_pair.citing_case
                ),
                "citing_case_year": str(case_pair.citing_case.year),
                "cited_case_name": format_case_name(case_pair.cited_case.case_name),
                "cited_case_citation": get_case_citation_for_scotus_case(
                    case_pair.cited_case
                ),
                "cited_case_year": str(case_pair.cited_case.year),
            },
            true_answer={
                "answer": str(int(case_pair.positive_relationship))
            },  # parsed as True/"yes" downstream
            correctness_callback=agreeement_correctness,
        )
        for case_pair in case_pairs
    ],
    sampling_temperature=1,
    save_string="scotus/doctrinal_agreement_few_shot",
)
doctrinal_agreement_task_few_shot.do()
doctrinal_agreement_task_few_shot.save()

###################################
# Overruled year task
###################################

# Load data
overruled_db: DataFrame = pd.read_csv(SCOTUS_OVERRULED_DB, index_col=False)
overruled_db = overruled_db[overruled_db.overruled_in_full == 1]
overruled_db = overruled_db.drop_duplicates(subset=["overruled_case_us_id"])

# Generate Case objects
overruled_cases: list[CourtCase] = [
    CourtCase(
        case_name=case["overruled_case_name"],
        us_citation=case["overruled_case_us_id"],
        year=case["overruled_case_year"],
        importance=0,
        majority_author=None,
        special_fact=str(case["year_overruled"]),
        court="scotus",
        source="overruled_db",
    )
    for case in overruled_db.to_dict("records")
]

# Task
year_overruled_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the year only.",
            query_template="What year was {case_name}, {case_citation}, overruled? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
            },
            true_answer={"answer": cast(str, case.special_fact)},
            correctness_callback=overruling_correctness,
            llm_answer_postprocess=clean_overruling_year,
        )
        for case in overruled_cases
    ],
    sampling_temperature=1,
    save_string="scotus/year_overruled",
)
year_overruled_task.do()
year_overruled_task.save()

# Task (few shot)
year_overruled_task_few_shot: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the year only.",
            query_template=inspect.cleandoc(
                """
				What year was the given case overruled? {system_message}

				Examples:
				```
				Case: Whitney v. California, 274 U.S. 357
				Answer: 1969

				Case: Austin v. Michigan Chamber of Commerce, 494 U.S. 652
				Answer: 2010
				```

				Case: {case_name}, {case_citation}
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": get_case_citation_for_scotus_case(case),
            },
            true_answer={"answer": cast(str, case.special_fact)},
            correctness_callback=overruling_correctness,
            llm_answer_postprocess=clean_overruling_year,
        )
        for case in overruled_cases
    ],
    sampling_temperature=1,
    save_string="scotus/year_overruled_few_shot",
)
year_overruled_task_few_shot.do()
year_overruled_task_few_shot.save()
