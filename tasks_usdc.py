import argparse
import inspect
from typing import cast

import pandas as pd
from pandas import DataFrame

from api import (
    GooglePaLMCompletion,
    LlamaChat,
    OpenAIChat,
    OpenAIChatGpt4,
    TogetherAiLlamaChat,
)
from correctness_checks import (
    bool_correctness,
    citation_correctness,
    cited_precedent_correctness,
    clean_district,
    clean_judge_name,
    clean_quotation,
    name_correctness,
    quotation_correctness,
    usdc_court_id_correctness,
)
from models import CourtCase, Query, Task
from settings import FSUPP_SAMPLE_PATH
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
usdc_sample: DataFrame = pd.read_csv(FSUPP_SAMPLE_PATH, index_col=False)

# Generate Case objects
cases: list[CourtCase] = [
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
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": "1"},  # parsed as True/"yes" downstream
            correctness_callback=bool_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/case_existence",
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
				Case: Scheck v. Burger King Corp., 756 F. Supp. 543 (1991)
				Answer: Yes

				Case: United States v. Apple Inc., 952 F. Supp. 2d 638 (2013)
				Answer: Yes

				Case: Columbia University v. Rodham, 564 F. Supp. 911 (1982)
				Answer: No
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": "1"},  # parsed as True/"yes" downstream
            correctness_callback=bool_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/case_existence_few_shot",
)
case_existence_task_few_shot.do()
case_existence_task_few_shot.save()

###################################
#  Court ID task
###################################
court_id_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message="Provide the name of the district court ONLY, nothing else.",
            query_template="Which federal district court decided the case {case_name}, {case_citation} ({case_year})? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.state)},
            correctness_callback=usdc_court_id_correctness,
            llm_answer_postprocess=clean_district,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/court_id",
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
            system_message="Provide the name of the district court ONLY, nothing else.",
            query_template=inspect.cleandoc(
                """
				Which federal district court decided the given case? {system_message}

				Examples:
				```
				Case: Scheck v. Burger King Corp., 756 F. Supp. 543 (1991)
				Answer: Southern District of Florida

				Case: United States v. Apple Inc., 952 F. Supp. 2d 638 (2013)
				Answer: Southern District of New York

				Case: United States v. Progressive, Inc., 467 F. Supp. 990 (1979)
				Answer: Western District of Wisconsin
				```

				Case name: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.state)},
            correctness_callback=usdc_court_id_correctness,
            llm_answer_postprocess=clean_district,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/court_id_few_shot",
)
court_id_task_few_shot.do()
court_id_task_few_shot.save()

###################################
#  Citation retrieval task
###################################
citation_retrieval_task: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Provide ONLY the citation in "<volume>, <reporter>, <page>" format, nothing else.',
            query_template="What is the citation for the district court case {case_name}? {system_message}",
            query_content={"case_name": format_case_name(case.case_name)},
            true_answer={"answer": cast(str, case.other_citation)},
            correctness_callback=citation_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/citation_retrieval",
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
				What is the citation for the given district court case? {system_message}

				Examples:
				```
				Case: Scheck v. Burger King Corp.
				Answer: 756 F. Supp. 543 (S.D. Fla. 1991)

				Case: United States v. Apple Inc.
				Answer: 952 F. Supp. 2d 638 (S.D.N.Y. 2013)

				Case: United States v. Progressive, Inc.
				Answer: 467 F. Supp. 990 (W.D. Wis. 1979)
				```

				Case: {case_name}
				Answer:
			"""
            ),
            query_content={"case_name": format_case_name(case.case_name)},
            true_answer={"answer": cast(str, case.other_citation)},
            correctness_callback=citation_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/citation_retrieval_few_shot",
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
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": clean_judge_name(cast(str, case.majority_author))},
            correctness_callback=name_correctness,
            llm_answer_postprocess=clean_judge_name,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/majority_author",
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
				Who wrote the opinion in the given case? {system_message}

				Examples:
				```
				Case: Scheck v. Burger King Corp., 756 F. Supp. 543 (1991)
				Answer: William Hoeveler

				Case: United States v. Apple Inc., 952 F. Supp. 2d 638 (2013)
				Answer: Denise Cote

				Case: United States v. Progressive, Inc., 467 F. Supp. 990 (1979)
				Answer: Robert Warren
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": clean_judge_name(cast(str, case.majority_author))},
            correctness_callback=name_correctness,
            llm_answer_postprocess=clean_judge_name,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="usdc/majority_author_few_shot",
)
majority_author_task_few_shot.do()
majority_author_task_few_shot.save()

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
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=quotation_correctness,
            llm_answer_postprocess=clean_quotation,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="usdc/quotation",
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
				Case: Scheck v. Burger King Corp., 756 F. Supp. 543 (1991)
				Answer: <quote>In cases in which jurisdiction depends upon diversity of citizenship, Federal courts must follow conflict of laws rules prevailing in the states in which they sit.</quote>

				Case: United States v. Apple Inc., 952 F. Supp. 2d 638 (2013)
				Answer: <quote>Another company's alleged violation of antitrust laws is not an excuse for engaging in your own violations of law.</quote>

				Case: United States v. Progressive, Inc., 467 F. Supp. 990 (1979)
				Answer: <quote>First Amendment rights are not absolute. They are not boundless.</quote>
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=quotation_correctness,
            llm_answer_postprocess=clean_quotation,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="usdc/quotation_few_shot",
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
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=cited_precedent_correctness,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="usdc/cited_precedent",
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
				What is a precedent that is cited in the opinion of the given case? {system_message}

				Examples:
				```
				Case: Scheck v. Burger King Corp., 756 F. Supp. 543 (1991)
				Answer: Klaxon Co. v. Stentor Electric Manufacturing Co., 313 U.S. 487

				Case: United States v. Apple Inc., 952 F. Supp. 2d 638 (2013)
				Answer: Anderson News, L.L.C. v. American Media, Inc., 680 F.3d 162

				Case: United States v. Progressive, Inc., 467 F. Supp. 990 (1979)
				Answer: New York Times v. United States, 403 U.S. 713
				```

				Case: {case_name}, {case_citation} ({case_year})
				Answer:
			"""
            ),
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": cast(str, case.majority_opinion)},
            correctness_callback=cited_precedent_correctness,
        )
        for case in cases
    ],
    sampling_temperature=-99,
    save_string="usdc/cited_precedent_few_shot",
)
cited_precedent_task_few_shot.do()
cited_precedent_task_few_shot.save()
