from __future__ import annotations

import ast
import csv
import inspect
import json
import re
import time
from abc import abstractmethod
from dataclasses import astuple, dataclass, field
from functools import partial
from glob import glob
from math import isinf
from typing import TYPE_CHECKING, Callable, ClassVar

import google
import google.ai.generativelanguage as safety_types
import google.generativeai as palm
import openai
import requests
from google.api_core.exceptions import InvalidArgument
from google.generativeai.types import BlockedReason
from manifest import Manifest
from openai.openai_object import OpenAIObject

manifest_instance = None

from reglab_secrets import (
    OPENAI_API_KEY,
    PALM_API_KEY1,
    PALM_API_KEY2,
    PALM_API_KEY3,
    PALM_API_KEY4,
    PALM_API_KEY5,
    TOGETHER_API_KEY,
)
from settings import RESULTS_SAVE_PATH
from utils import noop

PALM_API_KEYS = [
    PALM_API_KEY1,
    PALM_API_KEY2,
    PALM_API_KEY3,
    PALM_API_KEY4,
    PALM_API_KEY5,
]

if TYPE_CHECKING:
    from utils import AnswerType, APIResponseObjectType

openai.api_key = OPENAI_API_KEY
palm.configure(api_key=PALM_API_KEYS[0])


@dataclass
class APIBackend:
    """Generic class representing an LLM API backend"""

    # API parameters
    prompt: str
    sampling_temperature: float
    sampling_n: int
    max_tokens: int

    # Request variables
    greedy_request_callable: Callable[..., APIResponseObjectType] = partial(noop)
    sampled_request_callable: Callable[..., APIResponseObjectType] = partial(noop)

    # Greedy LLM response variables
    greedy_llm_answer: AnswerType = field(default_factory=dict)
    greedy_llm_answer_raw: str = ""
    greedy_response_object: APIResponseObjectType = field(default_factory=object)

    # Sampled LLM response variables
    sampled_llm_answers: list[AnswerType] = field(default_factory=list)
    sampled_llm_answers_raw: list[str] = field(default_factory=list)
    sampled_response_object: APIResponseObjectType = field(default_factory=object)

    # Token logging
    input_tokens: int = 0
    output_tokens: int = 0

    # Key rotation
    api_key_index: int = 0

    @abstractmethod
    def do_request(self) -> None:
        pass

    def retry(
        self, callable: Callable[..., APIResponseObjectType], try_n: int = 0
    ) -> APIResponseObjectType:
        try:
            return callable()
        except (
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            openai.error.APIConnectionError,
            google.api_core.exceptions.ResourceExhausted,
            google.api_core.exceptions.ServiceUnavailable,
        ) as e:
            # If a rate limit error occurs, first rotate API keys (only PaLM)
            # Error can happen even if we don't exceed our own limit
            # (e.g., the endpoint itself can be overloaded with too many requests)
            self.api_key_index = (self.api_key_index + 1) % len(PALM_API_KEYS)
            palm.configure(api_key=PALM_API_KEYS[self.api_key_index])

            # Next, re-try using an exponential backoff
            retry_time: int = 5 + try_n**2
            print(f"API ERROR: {e}; retrying in {retry_time} seconds.")
            time.sleep(retry_time)
            return self.retry(callable, try_n=try_n + 1)


@dataclass
class OpenAIChat(APIBackend):
    """Normal OpenAI ChatGPT-3.5 backend"""

    system_message: str | None = None

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        return partial(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": self.system_message or ""},
                {"role": "user", "content": self.prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=n,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            request_timeout=30,  # Note that this is currently undocumented
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(
                temperature=0, n=1, max_tokens=self.max_tokens
            )
            self.greedy_response_object = self.retry(self.greedy_request_callable)
            self.greedy_llm_answer_raw = self.greedy_response_object.choices[
                0
            ].message.content

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature,
                n=self.sampling_n,
                max_tokens=self.max_tokens,
            )
            self.sampled_response_object = self.retry(self.sampled_request_callable)
            self.sampled_llm_answers_raw = [
                choice.message.content
                for choice in self.sampled_response_object.choices
            ]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage
        if hasattr(self.greedy_response_object, "usage"):
            self.input_tokens += self.greedy_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.greedy_response_object["usage"][
                "completion_tokens"
            ]
        if hasattr(self.sampled_response_object, "usage"):
            self.input_tokens += self.sampled_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.sampled_response_object["usage"][
                "completion_tokens"
            ]


@dataclass
class OpenAIChatGpt4(APIBackend):
    """Normal OpenAI ChatGPT-3.5 backend"""

    system_message: str | None = None

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        return partial(
            openai.ChatCompletion.create,
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": self.system_message or ""},
                {"role": "user", "content": self.prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=n,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            request_timeout=30,  # Note that this is currently undocumented
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(
                temperature=0, n=1, max_tokens=self.max_tokens
            )
            self.greedy_response_object = self.retry(self.greedy_request_callable)
            self.greedy_llm_answer_raw = self.greedy_response_object.choices[
                0
            ].message.content

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature,
                n=self.sampling_n,
                max_tokens=self.max_tokens,
            )
            self.sampled_response_object = self.retry(self.sampled_request_callable)
            self.sampled_llm_answers_raw = [
                choice.message.content
                for choice in self.sampled_response_object.choices
            ]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage
        if hasattr(self.greedy_response_object, "usage"):
            self.input_tokens += self.greedy_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.greedy_response_object["usage"][
                "completion_tokens"
            ]
        if hasattr(self.sampled_response_object, "usage"):
            self.input_tokens += self.sampled_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.sampled_response_object["usage"][
                "completion_tokens"
            ]


@dataclass
class OpenAIChatJSON(APIBackend):
    """
    OpenAI ChatGPT-3.5 backend tricked into returning well-formatted JSON
    See docs: https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
    """

    functions: list[dict[str, str | dict]] = field(default_factory=list)
    function_call: dict[str, str] = field(default_factory=dict)
    system_message: str | None = None

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        return partial(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": self.system_message or ""},
                {"role": "user", "content": self.prompt},
            ],
            functions=self.functions,
            function_call=self.function_call,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=n,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            request_timeout=30,  # Note that this is currently undocumented
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(
                temperature=0, n=1, max_tokens=self.max_tokens
            )
            self.greedy_response_object = self.retry(self.greedy_request_callable)
            self.greedy_llm_answer_raw = self.greedy_response_object.choices[
                0
            ].message.function_call.arguments

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature,
                n=self.sampling_n,
                max_tokens=self.max_tokens,
            )
            self.sampled_response_object = self.retry(self.sampled_request_callable)
            self.sampled_llm_answers_raw = [
                choice.message.function_call.arguments
                for choice in self.sampled_response_object.choices
            ]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = json.loads(self.greedy_llm_answer_raw)
        self.sampled_llm_answers = [json.loads(s) for s in self.sampled_llm_answers_raw]

        # Record token usage
        if hasattr(self.greedy_response_object, "usage"):
            self.input_tokens += self.greedy_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.greedy_response_object["usage"][
                "completion_tokens"
            ]
        if hasattr(self.sampled_response_object, "usage"):
            self.input_tokens += self.sampled_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.sampled_response_object["usage"][
                "completion_tokens"
            ]


@dataclass
class OpenAICompletion(APIBackend):
    """Normal OpenAI GPT-3.5 backend"""

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        return partial(
            openai.Completion.create,
            model="text-davinci-003",
            prompt=self.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=n,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=1,
            request_timeout=30,  # Note that this is currently undocumented
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(
                temperature=0, n=1, max_tokens=int(self.max_tokens / self.sampling_n)
            )
            self.greedy_response_object = self.retry(self.greedy_request_callable)
            self.greedy_llm_answer_raw = self.greedy_response_object.choices[0].text

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature,
                n=self.sampling_n,
                max_tokens=self.max_tokens,
            )
            self.sampled_response_object = self.retry(self.sampled_request_callable)
            self.sampled_llm_answers_raw = [
                choice.text for choice in self.sampled_response_object.choices
            ]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage
        if hasattr(self.greedy_response_object, "usage"):
            self.input_tokens += self.greedy_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.greedy_response_object["usage"][
                "completion_tokens"
            ]
        if hasattr(self.sampled_response_object, "usage"):
            self.input_tokens += self.sampled_response_object["usage"]["prompt_tokens"]
            self.output_tokens += self.sampled_response_object["usage"][
                "completion_tokens"
            ]


@dataclass
class OpenAILogicalCheck(APIBackend):
    """OpenAI ChatGPT-3.5 backend for assessing logical compatibility or contradiction"""

    followup_prompt: str = ""

    def do_request(self) -> None:
        pass

    def _request1(self) -> dict:
        response1_object: OpenAIObject = self.retry(
            partial(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0,
                max_tokens=self.max_tokens,
                request_timeout=45,  # Note that this is currently undocumented
            )
        )
        return dict(response1_object.choices[0]["message"])

    def _request2(
        self, response1: dict
    ) -> tuple[str, Callable[..., APIResponseObjectType]]:
        request2_callable: Callable[..., APIResponseObjectType] = partial(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.prompt},
                response1,
                {"role": "user", "content": self.followup_prompt},
            ],
            temperature=0,
            max_tokens=5,  # should just be "yes" or "no"
            request_timeout=45,  # Note that this is currently undocumented
        )
        request2_response_object: OpenAIObject = self.retry(request2_callable)

        return (
            request2_response_object.choices[0]["message"]["content"],
            request2_callable,
        )

    def do_request_and_return(self) -> tuple[str, Callable[..., APIResponseObjectType]]:
        return self._request2(self._request1())


@dataclass
class OpenAIGpt4LogicalCheck(APIBackend):
    """OpenAI GPT4 backend for assessing logical compatibility or contradiction"""

    followup_prompt: str = ""

    def do_request(self) -> None:
        pass

    def _request1(self) -> tuple[str, Callable[..., APIResponseObjectType]]:
        callable_ = partial(
            openai.ChatCompletion.create,
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": self.prompt}],
            temperature=0,
            max_tokens=self.max_tokens,
            request_timeout=45,  # Note that this is currently undocumented
        )
        response1_object: OpenAIObject = self.retry(callable_)
        return (response1_object.choices[0]["message"]["content"], callable_)

    def do_request_and_return(self) -> tuple[str, Callable[..., APIResponseObjectType]]:
        return self._request1()


@dataclass
class OpenAIOnePassLogicalCheck(APIBackend):
    """OpenAI backend for assessing logical compatibility or contradiction"""

    followup_prompt: str = ""

    def do_request(self) -> None:
        pass

    def _request1(self) -> tuple[str, Callable[..., APIResponseObjectType]]:
        callable_ = partial(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": self.prompt}],
            temperature=0,
            max_tokens=self.max_tokens,
            request_timeout=45,  # Note that this is currently undocumented
        )
        response1_object: OpenAIObject = self.retry(callable_)
        return (response1_object.choices[0]["message"]["content"], callable_)

    def do_request_and_return(self) -> tuple[str, Callable[..., APIResponseObjectType]]:
        return self._request1()


@dataclass
class GooglePaLMChat(APIBackend):
    """Normal Google PaLM 2 chat backend"""

    def _build_request(self, temperature, n) -> Callable[..., APIResponseObjectType]:
        return partial(
            palm.chat, messages=self.prompt, temperature=temperature, candidate_count=n
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(temperature=0, n=1)
            self.greedy_response_object = self.retry(self.greedy_request_callable)
            try:
                self.greedy_llm_answer_raw = self.greedy_response_object.candidates[0][
                    "content"
                ]
            except IndexError:
                assert (
                    type(self.greedy_response_object.filters[0]["reason"])
                    is BlockedReason
                )
                self.greedy_llm_answer_raw = ""

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature, n=self.sampling_n
            )
            self.sampled_response_object = self.retry(self.sampled_request_callable)
            try:
                self.sampled_llm_answers_raw = [
                    candidate["content"]
                    for candidate in self.sampled_response_object.candidates
                ]
            except IndexError:
                assert (
                    type(self.sampled_response_object.filters[0]["reason"])
                    is BlockedReason
                )
                self.sampled_llm_answers_raw = []

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage (PaLM does not currently support)
        self.input_tokens = -99
        self.output_tokens = -99


@dataclass
class GooglePaLMCompletion(APIBackend):
    """Normal Google PaLM 2 completion backend"""

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        return partial(
            palm.generate_text,
            model="models/text-bison-001",
            max_output_tokens=max_tokens,
            prompt=self.prompt,
            temperature=temperature,
            candidate_count=n,
            safety_settings=[
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                    "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                },
            ],
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            try:
                self.greedy_request_callable = self._build_request(
                    temperature=0, n=1, max_tokens=self.max_tokens
                )
                self.greedy_response_object = self.retry(self.greedy_request_callable)
                self.greedy_llm_answer_raw = self.greedy_response_object.candidates[0][
                    "output"
                ]
            except IndexError:
                assert (
                    type(self.greedy_response_object.filters[0]["reason"])
                    is BlockedReason
                )
                self.greedy_llm_answer_raw = ""
            except (
                InvalidArgument
            ) as e:  # Weird bug where API tries to apply content filter to purportedly non-English query
                print(f"WARNING: {e}; prompt: {self.prompt}")
                self.greedy_llm_answer_raw = "-99"

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            try:
                self.sampled_request_callable = self._build_request(
                    temperature=self.sampling_temperature,
                    n=self.sampling_n,
                    max_tokens=self.max_tokens,
                )
                self.sampled_response_object = self.retry(self.sampled_request_callable)
                self.sampled_llm_answers_raw = [
                    candidate["output"]
                    for candidate in self.sampled_response_object.candidates
                ]
            except IndexError:
                assert (
                    type(self.sampled_response_object.filters[0]["reason"])
                    is BlockedReason
                )
                self.sampled_llm_answers_raw = []
            except (
                InvalidArgument
            ) as e:  # Weird bug where API tries to apply content filter to purportedly non-English query
                print(f"WARNING: {e}; prompt: {self.prompt}")
                self.sampled_llm_answers_raw = ["-99"]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage (PaLM does not currently support)
        self.input_tokens = -99
        self.output_tokens = -99


@dataclass
class LlamaChat(APIBackend):
    """LlamaChat backend"""

    system_message: str | None = None

    def __post_init__(self) -> None:
        # Must start the local huggingface model server first using Manifest:
        # python3 -m manifest.api.app --model_type huggingface --model_name_or_path /media/volume/sdb/huggingface/Llama-2-13b-chat-hf/ --model_generation_type text-generation --device 0
        global manifest_instance
        if manifest_instance is None:
            manifest_instance = Manifest(
                client_name="huggingface",
                client_connection="http://127.0.0.1:5000",
            )

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        assert manifest_instance
        return partial(
            manifest_instance.run,
            max_tokens=max_tokens,
            prompt=inspect.cleandoc(
                f"""
				<s>[INST] <<SYS>>
				{self.system_message}
				<</SYS>>

				{self.prompt} [/INST]
			"""
            ),
            temperature=temperature,
            n=n,
            return_response=True,
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(
                temperature=0.001, n=1, max_tokens=self.max_tokens
            )
            try:
                self.greedy_response_object = (
                    self.greedy_request_callable().get_response_obj()
                )
            except (
                requests.exceptions.HTTPError
            ) as e:  # FIXME: CUDA sometimes runs out of memory (maybe if the input prompt is too big??)
                print(e)
                self.greedy_llm_answer_raw = ""
            else:
                self.greedy_llm_answer_raw = self.greedy_response_object.choices[0].text

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature,
                n=self.sampling_n,
                max_tokens=self.max_tokens,
            )
            try:
                self.sampled_response_object = (
                    self.sampled_request_callable().get_response_obj()
                )
            except (
                requests.exceptions.HTTPError
            ) as e:  # FIXME: CUDA sometimes runs out of memory (maybe if the input prompt is too big??)
                print(e)
                self.sampled_llm_answers_raw = []
            else:
                self.sampled_llm_answers_raw = [
                    choice.text for choice in self.sampled_response_object.choices
                ]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage
        self.input_tokens = -99
        if hasattr(self.greedy_response_object, "choices"):
            self.output_tokens += len(
                [
                    tlp
                    for c in [
                        choice.token_logprobs
                        for choice in self.greedy_response_object.choices
                    ]
                    for tlp in c
                    if not isinf(tlp)
                ]
            )
        if hasattr(self.sampled_response_object, "choices"):
            self.output_tokens += len(
                [
                    tlp
                    for c in [
                        choice.token_logprobs
                        for choice in self.sampled_response_object.choices
                    ]
                    for tlp in c
                    if not isinf(tlp)
                ]
            )


@dataclass
class TogetherAiLlamaChat(APIBackend):
    """LlamaChat backend"""

    system_message: str | None = None

    @staticmethod
    def _together_ai_api_call(**kwargs):
        n = kwargs.pop("n", 1)
        if kwargs["temperature"] == 1:
            kwargs["temperature"] = 0.999
        response_json = None
        for i in range(n):
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                json=kwargs,
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json",
                },
            ).json()
            if response_json is None:
                response_json = response
            else:
                response_json["choices"].extend(response["choices"])
        response_object = openai.openai_object.OpenAIObject.construct_from(
            response_json
        )
        return response_object

    def _build_request(
        self, temperature, n, max_tokens
    ) -> Callable[..., APIResponseObjectType]:
        return partial(
            TogetherAiLlamaChat._together_ai_api_call,
            model="togethercomputer/llama-2-13b-chat",
            messages=[
                {"role": "system", "content": self.system_message or ""},
                {"role": "user", "content": self.prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            n=n,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    def do_request(self) -> None:
        # Only solicit a greedy response if this is not a zero-resource task
        if self.sampling_n != 2:
            self.greedy_request_callable = self._build_request(
                temperature=0.001, n=1, max_tokens=self.max_tokens
            )
            try:
                self.greedy_response_object = self.greedy_request_callable()
            except (
                requests.exceptions.HTTPError
            ) as e:  # FIXME: CUDA sometimes runs out of memory (maybe if the input prompt is too big??)
                print(e)
                self.greedy_llm_answer_raw = ""
            else:
                self.greedy_llm_answer_raw = self.greedy_response_object.choices[
                    0
                ].message.content

        # Only solicit sampled responses if temperature is set
        if self.sampling_temperature != -99:
            self.sampled_request_callable = self._build_request(
                temperature=self.sampling_temperature,
                n=self.sampling_n,
                max_tokens=self.max_tokens,
            )
            try:
                self.sampled_response_object = self.sampled_request_callable()
            except (
                requests.exceptions.HTTPError
            ) as e:  # FIXME: CUDA sometimes runs out of memory (maybe if the input prompt is too big??)
                print(e)
                self.sampled_llm_answers_raw = []
            else:
                self.sampled_llm_answers_raw = [
                    choice.message.content
                    for choice in self.sampled_response_object.choices
                ]

        # Reshape raw LLM answers into JSON
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]

        # Record token usage
        self.input_tokens = -99
        # no output token record


@dataclass
class DummyAPIBackend(APIBackend):
    """
    The DummyAPIBackend is a generic class for a backend that
    pulls inferences from existing results rather than querying any model.
    """

    @dataclass
    class APIInput:
        prompt: str
        sampling_temperature: float

    @dataclass
    class DummyRequestCallable:
        keywords: str

    @dataclass
    class StoredResult:
        api_input: DummyAPIBackend.APIInput
        greedy_llm_answer_raw: str
        sampled_llm_answers_raw: list[str]
        input_tokens: str
        output_tokens: str
        greedy_request_callable: DummyAPIBackend.DummyRequestCallable
        sampled_request_callable: DummyAPIBackend.DummyRequestCallable

    @staticmethod
    @abstractmethod
    def original_api_backend_type():
        pass

    stored_query_results: ClassVar[dict[tuple, StoredResult]] = {}

    @classmethod
    def load_stored_query_results(cls):
        original_api_backend_stored_files = glob(
            f"{RESULTS_SAVE_PATH}/*/*_{cls.original_api_backend_type().__name__}*.csv"
        )

        for file in original_api_backend_stored_files:
            if "state/" in file:
                continue
            sampling_temperature = float(re.search(r"=(-?\d+).csv", file).group(1))
            with open(file, "r") as csvfile:
                for line in csv.DictReader(csvfile):
                    # we read the post-processed results are stored in un-raw,
                    # and we want to redo all the processing as part of the dummy
                    # process
                    greedy_llm_answer_raw = line["llm_greedy_answer_raw"]
                    # we can't use JSON.loads b/c there are single quotes
                    # and we can't replace the single quotes because the
                    # answers could have apostrophes
                    sampled_llm_answers_raw = ast.literal_eval(
                        line["llm_sampled_answers_raw"]
                    )

                    prompt = line["query"]
                    sampling_n = len(sampled_llm_answers_raw)
                    api_input = DummyAPIBackend.APIInput(
                        prompt=prompt,
                        sampling_temperature=sampling_temperature,
                    )
                    if astuple(api_input) in cls.stored_query_results:
                        pass
                    stored_result = DummyAPIBackend.StoredResult(
                        api_input=api_input,
                        greedy_llm_answer_raw=greedy_llm_answer_raw,
                        sampled_llm_answers_raw=sampled_llm_answers_raw,
                        input_tokens=line["input_tokens"],
                        output_tokens=line["output_tokens"],
                        greedy_request_callable=DummyAPIBackend.DummyRequestCallable(
                            keywords=line["greedy_api_call"]
                        ),
                        sampled_request_callable=DummyAPIBackend.DummyRequestCallable(
                            keywords=line["sampled_api_call"]
                        ),
                    )
                    cls.stored_query_results[astuple(api_input)] = stored_result

    def __post_init__(self):
        if not self.__class__.stored_query_results:
            self.__class__.load_stored_query_results()

    def do_request(self) -> None:
        api_input = DummyAPIBackend.APIInput(
            prompt=self.prompt,
            sampling_temperature=self.sampling_temperature,
        )

        try:
            stored_query_result = self.__class__.stored_query_results[
                astuple(api_input)
            ]
        except KeyError:
            raise ValueError("No stored query result for {api_input}")

        if self.sampling_n != len(stored_query_result.sampled_llm_answers_raw):
            # print("Mismatch in sampling_n!")
            # There is one line with a mismatch because of a failed query.
            # needs a manual fix.
            pass
        self.greedy_llm_answer_raw = stored_query_result.greedy_llm_answer_raw
        self.sampled_llm_answers_raw = stored_query_result.sampled_llm_answers_raw
        self.greedy_llm_answer = {"answer": self.greedy_llm_answer_raw}
        self.sampled_llm_answers = [{"answer": s} for s in self.sampled_llm_answers_raw]
        self.input_tokens = int(stored_query_result.input_tokens)
        self.output_tokens = int(stored_query_result.output_tokens)
        self.greedy_request_callable = stored_query_result.greedy_request_callable  # type: ignore
        self.sampled_request_callable = stored_query_result.sampled_request_callable  # type: ignore


@dataclass
class DummyLlamaChat(DummyAPIBackend):
    def original_api_backend_type(self):
        return LlamaChat


@dataclass
class DummyOpenAIChat(DummyAPIBackend):
    def original_api_backend_type(self):
        return OpenAIChat


@dataclass
class DummyGooglePaLMCompletion(DummyAPIBackend):
    def original_api_backend_type(self):
        return GooglePaLMCompletion
