from __future__ import annotations

import copy
import csv
import os
import pickle
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BufferedWriter
from typing import TYPE_CHECKING, Any, Callable

import api
from correctness_checks import bool_correctness, clean_simple
from settings import NUM_THREADS, OBJECTS_SAVE_PATH, RESULTS_SAVE_PATH
from utils import agreement_cutoff, get_greedy_log_probs, print_progress

if TYPE_CHECKING:
    from utils import AnswerType, APIBackendType, APIResponseObjectType


@dataclass
class TestCase:
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class State(TestCase):
    def __init__(self, attr_dict: dict) -> None:
        for key, value in attr_dict.items():
            setattr(self, key.replace(" ", "_").replace("/", "_"), value)

        setattr(self, "source", "lsc")
        setattr(self, "state_name", getattr(self, "Jurisdictions"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_name": getattr(self, "state_name"),
            "source": getattr(self, "source"),
        }


@dataclass
class CourtCase(TestCase):
    case_name: str
    court: int | str
    importance: float
    majority_author: int | str | None
    source: str
    year: int
    majority_opinion: str | None = None
    state: str | None = None
    disposition: int | None = None
    winner: int | None = None
    lexis_citation: str | None = None
    sct_citation: str | None = None
    other_citation: str | None = None
    us_citation: str | None = None
    special_fact: str | None = None


@dataclass
class CourtCasePair(TestCase):
    citing_case: CourtCase
    cited_case: CourtCase
    positive_relationship: bool  # True=positive; False=negative


@dataclass
class Query:
    # Core query args
    test_case: TestCase
    query_template: str
    query_content: dict[str, str]
    true_answer: AnswerType | None = None  # If None, then this is a zero-resource query
    correctness_callback: Callable[[AnswerType, AnswerType], float] = bool_correctness
    answer_format: dict[str, dict[str, str | list]] | None = None
    llm_answer_postprocess: Callable[[str], str] = clean_simple
    query: str = ""
    system_message: str | None = None

    # API backend
    api_backend: api.APIBackend = field(init=False)

    # Correctness variables
    correctness: float = 0.0
    confidence: float = 0.0
    mean_logprobs: float = 0.0
    logical_check: Callable[
        ..., APIResponseObjectType
    ] | None = None  # Only for zero-resource tasks
    logical_check_response_content: str | None = None  # Only for zero-resource tasks

    # Logging
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        assert all(self.query_content.values())
        self.query = self.query_template.format(
            **{**self.query_content, "system_message": self.system_message}
        )

    def create_function_signature(self) -> dict:
        assert self.answer_format
        return {
            "type": "object",
            "properties": {k: v for k, v in self.answer_format.items()},
            "required": [k for k in self.answer_format.keys()],
        }

    def do_query(
        self,
        api_backend_type: APIBackendType,
        sampling_temperature: float,
        max_tokens: int,
    ) -> None:
        # Call API
        match api_backend_type:
            case api.OpenAIChat:
                self.api_backend = api.OpenAIChat(
                    prompt=self.query,
                    system_message=self.system_message,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.OpenAIChatGpt4:
                self.api_backend = api.OpenAIChatGpt4(
                    prompt=self.query,
                    system_message=self.system_message,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.OpenAICompletion:
                self.api_backend = api.OpenAICompletion(
                    prompt=self.query,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.OpenAIChatJSON:
                self.api_backend = api.OpenAIChatJSON(
                    prompt=self.query,
                    system_message=self.system_message,
                    functions=[
                        {
                            "name": "answer_query",
                            "parameters": self.create_function_signature(),
                        }
                    ],
                    function_call={"name": "answer_query"},
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.GooglePaLMChat:
                self.api_backend = api.GooglePaLMChat(
                    prompt=self.query,
                    sampling_temperature=sampling_temperature,
                    sampling_n=8 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.GooglePaLMCompletion:
                self.api_backend = api.GooglePaLMCompletion(
                    prompt=self.query,
                    sampling_temperature=sampling_temperature,
                    sampling_n=8 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.LlamaChat:
                self.api_backend = api.LlamaChat(
                    prompt=self.query,
                    system_message=self.system_message,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.TogetherAiLlamaChat:
                self.api_backend = api.TogetherAiLlamaChat(
                    prompt=self.query,
                    system_message=self.system_message,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.DummyLlamaChat:
                self.api_backend = api.DummyLlamaChat(
                    prompt=self.query,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.DummyOpenAIChat:
                self.api_backend = api.DummyOpenAIChat(
                    prompt=self.query,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case api.DummyGooglePaLMCompletion:
                self.api_backend = api.DummyGooglePaLMCompletion(
                    prompt=self.query,
                    sampling_temperature=sampling_temperature,
                    sampling_n=10 if self.true_answer else 2,
                    max_tokens=max_tokens,
                )
            case _:
                raise Exception("Invalid API backend.")
        self.api_backend.do_request()

        # Post-process LLM response
        self.api_backend.greedy_llm_answer = {
            k: self.llm_answer_postprocess(v)
            for k, v in self.api_backend.greedy_llm_answer.items()
        }
        self.api_backend.sampled_llm_answers = [
            {k: self.llm_answer_postprocess(v) for k, v in a.items()}
            for a in self.api_backend.sampled_llm_answers
        ]

        # Calculate correctness and confidence scores
        # CASE 1: Zero-resource task (there's no true_answer, so we just check for contradiction between the two sampled answers)
        if not self.true_answer:
            if "-99" in self.api_backend.sampled_llm_answers:
                # If the model declined to generate one of the sampled answers, correctness cannot be calculated
                self.correctness = -99
            if len(self.api_backend.sampled_llm_answers) < 2:
                # If only one sampled answer is present (only possible in PaLM since it removes duplicates),
                # the originally produced answers are literally identical and thus non-contradictory
                self.correctness = 100.0
            else:
                # For zero-resource tasks, we override the correctness_callback() function in kind of a hacky way to return BOTH a float
                # representing correctness (like normal) AND a reference to the logical check callable, which we need to save for logging
                # purposes.
                self.correctness_callback = partial(
                    self.correctness_callback, query_content=self.query_content
                )
                self.correctness, self.logical_check_response_content, self.logical_check = self.correctness_callback(self.api_backend.sampled_llm_answers[0], self.api_backend.sampled_llm_answers[1])  # type: ignore

        # CASE 2: Resource-aware task (we can assess correctness directly and calculate calibration across the 8 or 10 sampled answers)
        else:
            self.correctness = self.correctness_callback(
                self.api_backend.greedy_llm_answer, self.true_answer
            )

            # Only calculate confidence if the model provides a greedy response and sampling is enabled
            if self.correctness != -99 and self.api_backend.sampled_llm_answers:
                # Save a copy of the correctness check to export later
                for a in self.api_backend.sampled_llm_answers:
                    a["greedy_agreement"] = str(
                        self.correctness_callback(a, self.api_backend.greedy_llm_answer)
                    )

                # Take the mean of the correctness scores across all samples
                self.confidence = statistics.fmean(
                    [
                        agreement_cutoff(a["greedy_agreement"])
                        for a in self.api_backend.sampled_llm_answers
                    ]
                )

                # If OpenAICompletion, also extract the token-level logprobs as an alternative metric
                if api_backend_type is api.OpenAICompletion:
                    self.mean_logprobs = get_greedy_log_probs(
                        self.api_backend.greedy_response_object
                    )

        # Log timestamp
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = self.test_case.to_dict()
        d.pop("majority_opinion", None)
        d["query"] = self.query
        d["llm_greedy_answer_raw"] = self.api_backend.greedy_llm_answer_raw
        d["llm_greedy_answer"] = str(self.api_backend.greedy_llm_answer)
        d["true_answer"] = (
            str(self.true_answer)[:100] + "...}'"
            if len(str(self.true_answer)) > 100
            else str(self.true_answer)
        )  # Truncate answers that are majority opinions
        d["llm_sampled_answers_raw"] = str(self.api_backend.sampled_llm_answers_raw)
        d["llm_sampled_answers"] = str(self.api_backend.sampled_llm_answers)
        d["correctness"] = self.correctness
        d["confidence"] = self.confidence
        d["mean_logprobs"] = self.mean_logprobs
        d["timestamp"] = self.timestamp
        d["input_tokens"] = self.api_backend.input_tokens
        d["output_tokens"] = self.api_backend.output_tokens
        d["greedy_api_call"] = str(self.api_backend.greedy_request_callable.keywords)  # type: ignore
        d["sampled_api_call"] = str(self.api_backend.sampled_request_callable.keywords)  # type: ignore
        d["logical_check_response_content"] = (
            self.logical_check_response_content
            if self.logical_check_response_content is not None
            else ""
        )
        d["logical_check_api_call"] = (
            str(self.logical_check.keywords)
            if self.logical_check is not None
            and hasattr(self.logical_check, "keywords")
            else ""
        )
        return d


@dataclass
class Task:
    queries: list[Query]
    save_string: str
    api_backend_type: APIBackendType = api.OpenAIChat
    sampling_temperature: float = 0.5
    max_tokens: int = 100

    def do(self) -> None:
        # Run queries in parallel
        self.completed: set = set()
        self.written: set = set()
        self.save_file = open(
            os.path.join(
                RESULTS_SAVE_PATH,
                f"{self.save_string}_{self.api_backend_type.__name__}_results_temp={self.sampling_temperature}.csv",
            ),
            "w",
        )
        self.writer: csv.DictWriter | None = None

        i_queries = list(enumerate(self.queries))

        def run_queries(i_queries):
            nonlocal self
            for index, query in i_queries:
                try:
                    query.do_query(
                        self.api_backend_type,
                        self.sampling_temperature,
                        self.max_tokens,
                    )
                except ValueError as e:
                    if "No stored query result" in str(e):
                        print(f"WARNING: Query failed, no stored result.")
                self.completed.add(index)
                print_progress(
                    len(self.completed), len(self.queries), prefix=self.save_string
                )

        def regular_export():
            while len(self.completed) < len(self.queries):
                self.export()
                time.sleep(5)

        threads = []
        for i in range(NUM_THREADS):
            t = threading.Thread(target=run_queries, args=(i_queries[i::NUM_THREADS],))
            threads.append(t)
            t.start()
        t = threading.Thread(target=regular_export)
        t.start()
        threads.append(t)
        for t in threads:
            t.join()

    def export(self) -> None:
        # Export case-wise results as .csv for downstream processing
        to_write = self.completed - self.written
        export: list[dict[str, Any]] = [self.queries[i].to_dict() for i in to_write]
        if not export:
            return
        if not self.__dict__.get("save_file", None):
            self.save_file = open(
                os.path.join(
                    RESULTS_SAVE_PATH,
                    f"{self.save_string}_{self.api_backend_type.__name__}_results_temp={self.sampling_temperature}.csv",
                ),
                "w",
            )
        if not self.__dict__.get("writer", None):
            self.writer = csv.DictWriter(
                self.save_file, fieldnames=list(export[0].keys())
            )
            self.writer.writeheader()
        self.writer.writerows(export)  # type: ignore
        self.written = self.written | to_write

    def save(self) -> None:
        # Delete majority opinion to save space
        for query in self.queries:
            if hasattr(query.test_case, "majority_opinion"):
                query.test_case = copy.deepcopy(query.test_case)
                query.test_case.majority_opinion = ""  # type: ignore

        # PaLM ChatResponse and Completion objects are not natively pickle-able, so convert them to dicts first
        if self.api_backend_type in [api.GooglePaLMChat, api.GooglePaLMCompletion]:
            for query in self.queries:
                if hasattr(query.api_backend.greedy_response_object, "to_dict"):
                    query.api_backend.greedy_response_object = (
                        query.api_backend.greedy_response_object.to_dict()
                    )
                if hasattr(query.api_backend.sampled_response_object, "to_dict"):
                    query.api_backend.sampled_response_object = (
                        query.api_backend.sampled_response_object.to_dict()
                    )

        # Save full object to disk
        _save_file = self.save_file
        del self.save_file
        _writer = self.writer
        del self.writer
        save_file: BufferedWriter = open(
            os.path.join(
                OBJECTS_SAVE_PATH,
                f"{self.save_string}_{self.api_backend_type.__name__}_{time.time()}.pickle",
            ),
            "wb",
        )
        pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)
        save_file.close()

        # Restore full object
        self.save_file = _save_file
        self.writer = _writer
        self.export()
        self.save_file.close()
