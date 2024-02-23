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
    bool_correctness,
    citation_correctness,
    cited_precedent_correctness,
    clean_circuit,
    clean_judge_name,
    clean_quotation,
    coa_court_id_correctness,
    name_correctness,
    quotation_correctness,
)
from models import CourtCase, Query, Task
from settings import FD_SAMPLE_PATH, SONGER_SAMPLE_PATH
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
cap_sample: DataFrame = pd.read_csv(FD_SAMPLE_PATH, index_col=False)

# Generate Case objects
cases: list[CourtCase] = [
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
    for case in cap_sample.to_dict("records")
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
    save_string="coa/case_existence",
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
				Case: Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2012)
				Answer: Yes

				Case: Bonner v. City of Prichard, Alabama, 661 F.2d 1206 (1981)
				Answer: Yes

				Case: Columbia University v. Rodham, 564 F.2d. 911 (1977)
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
    save_string="coa/case_existence_few_shot",
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
            system_message="Provide the name of the circuit ONLY, nothing else.",
            query_template="Which federal circuit court decided the case {case_name}, {case_citation} ({case_year})? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": str(case.court)},
            correctness_callback=coa_court_id_correctness,
            llm_answer_postprocess=clean_circuit,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="coa/court_id",
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
            system_message="Provide the name of the circuit ONLY, nothing else.",
            query_template=inspect.cleandoc(
                """
				Which federal circuit court decided the given case? {system_message}

				Examples:
				```
				Case: Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2012)
				Answer: Second Circuit

				Case: Durham v. United States, 214 F.2d 862 (1954)
				Answer: D.C. Circuit

				Case: Bonner v. City of Prichard, Alabama, 661 F.2d 1206 (1981)
				Answer: Eleventh Circuit
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
            true_answer={"answer": str(case.court)},
            correctness_callback=coa_court_id_correctness,
            llm_answer_postprocess=clean_circuit,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="coa/court_id_few_shot",
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
            query_template="What is the citation for the circuit court case {case_name}? {system_message}",
            query_content={"case_name": format_case_name(case.case_name)},
            true_answer={"answer": cast(str, case.other_citation)},
            correctness_callback=citation_correctness,
        )
        for case in cases
    ],
    sampling_temperature=1,
    save_string="coa/citation_retrieval",
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
				What is the citation for the given circuit court case? {system_message}

				Examples:
				```
				Case: Viacom International Inc. v. YouTube, Inc.
				Answer: 676 F.3d 19

				Case: United States v. One Book Called Ulysses
				Answer: 72 F.2d 705

				Case: Bonner v. City of Prichard, Alabama
				Answer: 661 F.2d 1206
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
    save_string="coa/citation_retrieval_few_shot",
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
    save_string="coa/majority_author",
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
				Case: Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2012)
				Answer: Jose Cabranes

				Case: Durham v. United States, 214 F.2d 862 (1954)
				Answer: David Bazelon

				Case: Bonner v. City of Prichard, Alabama, 661 F.2d 1206 (1981)
				Answer: John Godbold
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
    save_string="coa/majority_author_few_shot",
)
majority_author_task_few_shot.do()
majority_author_task_few_shot.save()

###################################
# Affirm/reverse task
###################################
# Load Songer data for this task
songer_sample: DataFrame = pd.read_csv(SONGER_SAMPLE_PATH, index_col=False)

# Generate Case objects
songer_cases: list[CourtCase] = [
    CourtCase(
        case_name=case["case_name"],
        other_citation=case["citation"],
        year=case["year"],
        majority_author="",
        court=case["circuit"],
        disposition=case["disposition"],
        source="songer",
        importance=0,
    )
    for case in songer_sample.to_dict("records")
]

# Task
task_affirm_reverse: Task = Task(
    api_backend_type=CURRENT_API,
    queries=[
        Query(
            test_case=case,
            system_message='Say "affirm" or "reverse" only.',
            query_template="Did the court in {case_name}, {case_citation} ({case_year}) affirm or reverse the lower court's decision? {system_message}",
            query_content={
                "case_name": format_case_name(case.case_name),
                "case_citation": cast(str, case.other_citation),
                "case_year": str(case.year),
            },
            true_answer={"answer": "affirm" if case.disposition == 1 else "reverse"},
            correctness_callback=affirm_reverse_correctness,
        )
        for case in songer_cases
    ],
    sampling_temperature=1,
    save_string="coa/affirm_reverse",
)
task_affirm_reverse.do()
task_affirm_reverse.save()

# Task (few shot)
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
				Case: United States v. One Book Called Ulysses, 72 F.2d 705 (1934)
				Answer: Reverse

				Case: Durham v. United States, 214 F.2d 862 (1954)
				Answer: Reverse

				Case: United States v. Blackley, 167 F.3d. 543 (1999)
				Answer: Affirm
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
            true_answer={"answer": "affirm" if case.disposition == 1 else "reverse"},
            correctness_callback=affirm_reverse_correctness,
        )
        for case in songer_cases
    ],
    sampling_temperature=1,
    save_string="coa/affirm_reverse_few_shot",
)
task_affirm_reverse_few_shot.do()
task_affirm_reverse_few_shot.save()

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
    save_string="coa/quotation",
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
				Case: United States v. One Book Called Ulysses, 72 F.2d 705 (1934)
				Answer: <quote>The book as a whole is not pornographic, and, while in not a few spots it is coarse, blasphemous, and obscene, it does not, in our opinion, tend to promote lust.</quote>

				Case: Durham v. United States, 214 F.2d 862 (1954)
				Answer: <quote>In attempting to define insanity in terms of a symptom, the courts have assumed an impossible role</quote>

				Case: Bonner v. City of Prichard, Alabama, 661 F.2d 1206 (1981)
				Answer: <quote>Failing to select a body of precedent would severely impede the operation of the court.</quote>
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
    save_string="coa/quotation_few_shot",
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
    save_string="coa/cited_precedent",
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
				Case: Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2012)
				Answer: Universal City Studios, Inc. v. Corley, 273 F.3d 429

				Case: United States v. One Book Called Ulysses, 72 F.2d 705 (1934)
				Answer: United States v. Dennett, 39 F.2d 564

				Case: Bonner v. City of Prichard, Alabama, 661 F.2d 1206 (1981)
				Answer: Moragne v. States Marine Lines, 398 U.S. 375
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
    save_string="coa/cited_precedent_few_shot",
)
cited_precedent_task_few_shot.do()
cited_precedent_task_few_shot.save()
