import ast
import csv
import enum
import json
import os
import re
import sqlite3
import sys
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peewee as pw
import seaborn as sns
from peewee import fn
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase
from tqdm import tqdm

from settings import FIGURES_SAVE_PATH, RANDOM_SEED, RESULTS_SAVE_PATH

db = SqliteExtDatabase(
    os.path.join(RESULTS_SAVE_PATH, "results.db"),
    pragmas=(
        ("cache_size", -1024 * 64),  # 64MB page-cache.
        ("journal_mode", "wal"),  # Use WAL-mode (you should always use this!).
        ("foreign_keys", 1),
    ),
)  # Enforce foreign-key constraints.
sns.set_theme("paper", style="white", font="Times New Roman")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{mathptmx}",
        "font.size": 11,
        "figure.figsize": (5, 2.5),
        "figure.dpi": 300,
    }
)
sns.set_palette("colorblind")  # , 8)

TASK_PATH_REGEX = re.compile(
    r"/(?:\w+/)+(?P<court_level>\w+)/(?P<task>(?:\w+_)+?)(?P<llm>\w+)_results_temp=(?P<temperature>-?(?:\d{1,2}|0.999))_?(?P<correctness_labeler>\w+)?\.csv"
)
FORMAT: str = "eps"
LLM_MODEL_NAMES = {
    "OpenAIChatGpt4": "GPT 4",
    "OpenAIChat": "GPT 3.5",
    "GooglePaLMCompletion": "PaLM 2",
    "LlamaChat": "Llama 2",
}


def _parse_task_path(path):
    match = re.match(TASK_PATH_REGEX, path)
    if not match:
        raise ValueError(f"{path} is not a valid task path")
    groups = match.groupdict()
    groups["prompt_style"] = "few_shot" if "few_shot" in groups["task"] else "zero_shot"
    groups["task"] = groups["task"].replace("few_shot", "").strip("_")
    groups["llm"] = LLM_MODEL_NAMES[groups["llm"]]
    groups["correctness_labeler"] = groups.get("correctness_labeler", "resource_aware")
    return groups


class CourtLevel(enum.StrEnum):
    COA = "coa"
    USDC = "usdc"
    SCOTUS = "scotus"


class Llm(enum.StrEnum):
    GPT4 = "GPT 4"
    CHATGPT = "GPT 3.5"
    PALM = "PaLM 2"
    LLAMA = "Llama 2"


class Task(enum.StrEnum):
    EXISTENCE = "case_existence"
    COURT = "court_id"
    CITATION = "citation_retrieval"
    AUTHOR = "majority_author"
    DISPOSITION = "affirm_reverse"
    OVERRULING_YEAR = "year_overruled"
    DOCTRINAL_AGREEMENT = "doctrinal_agreement"
    QUOTATION = "quotation"
    POSTURE = "posture"
    SUBSEQUENT_HISTORY = "subsequent_history"
    FACTUAL_BACKGROUND = "factual_background"
    HOLDING = "holding"
    FAKE_YEAR = "fake_year"
    FAKE_CASE_EXISTENCE = "fake_case_existence"
    FAKE_DISSENT = "fake_dissent"
    AUTHORITY = "cited_precedent"
    CORE_LEGAL_QUESTION = "core_legal_question"


LOW_COMPLEXITY_TASKS = [Task.EXISTENCE, Task.COURT, Task.CITATION, Task.AUTHOR]
MEDIUM_COMPLEXITY_TASKS = [
    Task.DISPOSITION,
    Task.OVERRULING_YEAR,
    Task.AUTHORITY,
    Task.QUOTATION,
]
HIGH_COMPLEXITY_TASKS = [
    Task.DOCTRINAL_AGREEMENT,
    Task.HOLDING,
    Task.FACTUAL_BACKGROUND,
    Task.POSTURE,
    Task.SUBSEQUENT_HISTORY,
    Task.CORE_LEGAL_QUESTION,
]
ZERO_RESOURCE_TASKS = [
    Task.HOLDING,
    Task.FACTUAL_BACKGROUND,
    Task.POSTURE,
    Task.SUBSEQUENT_HISTORY,
    Task.CORE_LEGAL_QUESTION,
]
ANCHORING_BIAS_TASKS = [Task.FAKE_YEAR, Task.FAKE_CASE_EXISTENCE, Task.FAKE_DISSENT]


def convert_json(dirty_json: str | None) -> str:
    if not dirty_json:
        return ""

    try:
        return json.dumps(json.loads(dirty_json))
    except:
        if dirty_json[-5:] == "...}'":
            if dirty_json[11] == '"':
                dirty_json = dirty_json[:-5] + '..."}'
            elif dirty_json[11] == "'":
                dirty_json = dirty_json[:-5] + "...'}"

        parsed_dict = ast.literal_eval(dirty_json)
        clean_json = json.dumps(parsed_dict)
        return clean_json


class PromptStyle(enum.StrEnum):
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"


class Case(pw.Model):
    source = pw.TextField()
    case_name = pw.TextField()
    court = pw.TextField(null=True)
    court_level = pw.TextField(choices=[(e.value, e.name) for e in CourtLevel])
    importance = pw.FloatField()
    majority_author = pw.TextField(null=True)
    year = pw.IntegerField()
    state = pw.TextField(null=True)
    disposition = pw.IntegerField(null=True)
    winner = pw.IntegerField(null=True)
    lexis_citation = pw.TextField(null=True)
    other_citation = pw.TextField(null=True)
    us_citation = pw.TextField(null=True)
    wikipedia_length = pw.IntegerField(null=True)

    class Meta:
        database = db
        indexes = (
            (("case_name", "us_citation", "other_citation", "importance"), True),
        )

    def citation(self):
        return self.us_citation or self.other_citation or self.lexis_citation

    @classmethod
    def _field_names(cls):
        return [
            k for k, v in cls._meta.fields.items() if not isinstance(v, pw.AutoField)
        ]

    def _key(self):
        fields = self.__class__._meta.indexes[0][0]
        return tuple((self.__getattribute__(field) for field in fields))

    @classmethod
    def load_wikipedia(cls):
        pass

    @classmethod
    def from_row(cls, metadata, row):
        return cls(
            **{k: row[k] or None for k in cls._field_names() if k in row}
            | {k: metadata[k] or None for k in cls._field_names() if k in metadata}
        )

    @classmethod
    def load(cls):
        try:
            cls.drop_table()
        except (sqlite3.IntegrityError, pw.IntegrityError):
            print(
                "ERROR: You have already run Case.load() once. If you want to reload, first delete the TaskRun table by running TaskRun.drop_table(). Then try running Case.load() again."
            )
            sys.exit(1)
        cls.create_table()
        cases = {}
        for case_ in tqdm(glob(f"{RESULTS_SAVE_PATH}/*/*.csv")):
            if "tasks/state" in case_:
                continue
            metadata = _parse_task_path(case_)
            if metadata["task"] == "doctrinal_agreement":
                continue
            with open(case_) as f:
                lines = list(csv.DictReader(f))
            in_this_file = set()
            for l in lines:
                c = cls.from_row(metadata, l)
                key = c._key()
                suffix = 2
                while key in in_this_file:
                    c.case_name += f" ({suffix})"
                    suffix += 1
                    key = c._key()
                if c._key() not in cases:
                    cases[(c._key())] = c
                in_this_file.add(c._key())
        to_create = list(cases.values())
        while to_create:
            cls.bulk_create(to_create[:500])
            to_create = to_create[500:]


class TaskRun(pw.Model):
    court_level = pw.TextField(choices=[(e.value, e.name) for e in CourtLevel])
    task = pw.TextField(choices=[(e.value, e.name) for e in Task])
    llm = pw.TextField(choices=[(e.value, e.name) for e in Llm])
    temperature = pw.IntegerField()
    prompt_style = pw.TextField(choices=[(e.value, e.name) for e in PromptStyle])
    case = pw.ForeignKeyField(Case, backref="taskruns")
    query = pw.TextField(null=True)
    llm_greedy_answer = pw.TextField(null=True)
    llm_greedy_answer_raw = JSONField(null=True)
    true_answer = pw.TextField(null=True)
    llm_sampled_answers = JSONField(null=True)
    llm_sampled_answers_raw = JSONField(null=True)
    correctness = pw.IntegerField()
    correctness_labeler = pw.TextField(null=True)
    confidence = pw.FloatField()

    @classmethod
    def _field_names(cls):
        return [
            k for k, v in cls._meta.fields.items() if not isinstance(v, pw.AutoField)
        ]

    def _key(self):
        fields = self.__class__._meta.indexes[0][0]
        return tuple((self.__getattribute__(field) for field in fields))

    class Meta:
        database = db
        indexes = (
            (
                (
                    "court_level",
                    "task",
                    "llm",
                    "prompt_style",
                    "case_id",
                    "correctness_labeler",
                ),
                True,
            ),
        )

    @classmethod
    def load(cls):
        cls.drop_table()
        cls.create_table()
        created = {}

        key_to_case = {c._key(): c for c in Case.select()}
        for case_ in tqdm(sorted(glob(f"{RESULTS_SAVE_PATH}/*/*.csv"))):
            if "tasks/state" in case_:
                continue
            if "TogetherAi" in case_:
                continue
            metadata = _parse_task_path(case_)
            if metadata["task"] == "doctrinal_agreement":
                continue
            with open(case_) as f:
                lines = list(csv.DictReader(f))
            cases_in_this_file = set()
            for l in lines:
                case = Case.from_row(metadata, l)
                case_key = case._key()
                suffix = 2
                while case_key in cases_in_this_file:
                    l["case_name"] += f" ({suffix})"
                    suffix += 1
                    case = Case.from_row(metadata, l)
                    case_key = case._key()
                cases_in_this_file.add(case_key)
                case_key = case_key[:-1] + (float(case_key[-1]),)
                if case_key not in key_to_case:
                    raise ValueError(f"Could not find case {case._key()}")
                case_id = key_to_case[case_key].id
                fields = {k: l[k] or None for k in cls._field_names() if k in l} | {
                    k: metadata[k] or None for k in cls._field_names() if k in metadata
                }
                if "llm_greedy_answer" in fields:
                    fields["llm_greedy_answer"] = convert_json(
                        fields["llm_greedy_answer"]
                    )
                if "true_answer" in fields:
                    fields["true_answer"] = convert_json(fields["true_answer"])
                c = cls(
                    case_id=case_id,
                    **fields,
                )
                if c._key() in created:
                    warnings.warn(f"Duplicate key {c._key()} in {case._key()}")
                created[c._key()] = c
        to_create = list(created.values())
        while to_create:
            cls.bulk_create(to_create[:500])
            to_create = to_create[500:]


_HALLUCINATION_RATE_WITHOUT_ABSTENTIONS = 1 - pw.fn.SUM(
    TaskRun.correctness > 72
) * 1.0 / pw.fn.SUM(TaskRun.correctness > -99)

HALLUCINATION_RATE = 1 - pw.fn.SUM(
    (TaskRun.correctness > 72) | (TaskRun.correctness == -99)
) * 1.0 / pw.fn.SUM(TaskRun.correctness > -100)


def _create_bins(group, column, num_bins=10):
    """Helper function to do binning."""
    np.random.seed(RANDOM_SEED)
    noise = np.random.normal(-1e-6, 1e-6, size=group[column].shape)
    group["bin"] = pd.qcut(group[column] + noise, q=num_bins, labels=False)
    return group


plot_methods = []


def plot_method(f):
    plot_methods.append(f)
    return f


@plot_method
def expected_calibration_error_per_llm():
    query = list(
        TaskRun.select(
            (TaskRun.confidence / 100).alias("confidence"),
            (TaskRun.correctness > 72).alias("correct"),
            TaskRun.llm.alias("llm"),
        )
        .where(
            (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.correctness > -99)
            & (TaskRun.temperature > -99)
        )
        .dicts()
    )
    df = pd.DataFrame.from_records(query)

    # Bin df
    df_binned = df.groupby("llm").apply(_create_bins, "confidence")
    df_binned = df_binned.reset_index(drop=True)

    # Generate bin-level means for each LLM
    bin_summaries = df_binned.groupby(["llm", "bin"]).agg(
        mean_confidence=pd.NamedAgg(column="confidence", aggfunc="mean"),
        mean_correct=pd.NamedAgg(column="correct", aggfunc="mean"),
    )
    bin_summaries = bin_summaries.reset_index()

    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="lightgrey", label="Ideal"
    )  # draw ideal line at y=x
    sns.lineplot(
        data=bin_summaries,
        x="mean_confidence",
        y="mean_correct",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        legend=False,
    )
    sns.scatterplot(
        data=bin_summaries,
        x="mean_confidence",
        y="mean_correct",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        style="llm",
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles=handles[1:], labels=labels[1:], frameon=False)
    plt.xlabel("Confidence")
    plt.ylabel("Non-Hallucination Rate")
    plt.ylim(0, 1)
    plt.xlim(-0.05, 1.05)
    plt.gcf().tight_layout()


@plot_method
def temp_scaled_expected_calibration_error_per_llm():
    import calibration as cal

    # Get data
    query = list(
        TaskRun.select(
            (TaskRun.confidence / 100).alias("confidence"),
            (TaskRun.correctness > 72).alias("correct"),
            TaskRun.llm.alias("llm"),
        )
        .where(
            (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.correctness > -99)
            & (TaskRun.temperature > -99)
        )
        .dicts()
    )
    df = pd.DataFrame.from_records(query)

    # Do temperature scaling
    for llm in Llm:
        data = df[df.llm == llm]
        calibrator = cal.PlattCalibrator(
            len(data), num_bins=10
        )  # Note: num_bins parameter must be passed, but is unused in this function
        calibrator.train_calibration(data["confidence"].values, data["correct"].values)
        df.loc[df["llm"] == llm, "scaled_confidence"] = calibrator.calibrate(
            data["confidence"].values
        )

    # Bin df
    df_binned = df.groupby("llm").apply(_create_bins, "scaled_confidence")
    df_binned = df_binned.reset_index(drop=True)

    # Generate bin-level means for each LLM
    bin_summaries = df_binned.groupby(["llm", "bin"]).agg(
        mean_confidence=pd.NamedAgg(column="scaled_confidence", aggfunc="mean"),
        mean_correct=pd.NamedAgg(column="correct", aggfunc="mean"),
    )
    bin_summaries = bin_summaries.reset_index()

    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="lightgrey", label="Ideal"
    )  # draw ideal line at y=x
    sns.lineplot(
        data=bin_summaries,
        x="mean_confidence",
        y="mean_correct",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        legend=False,
    )
    if True:
        sns.scatterplot(
            data=bin_summaries,
            x="mean_confidence",
            y="mean_correct",
            hue="llm",
            hue_order=[llm.value for llm in Llm],
        )
    else:
        sns.jointplot(
            data=bin_summaries,
            x="mean_confidence",
            y="mean_correct",
            hue="llm",
            hue_order=[llm.value for llm in Llm],
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles=handles[1:], labels=labels[1:], frameon=False)
    plt.xlabel("Confidence")
    plt.ylabel("Non-Hallucination Rate")
    plt.ylim(0, 1)
    plt.xlim(-0.05, 1.05)
    plt.gcf().tight_layout()

    # Print out ECE comparison
    for llm in Llm:
        unscaled_ece = cal.get_ece(
            df[df.llm == llm]["confidence"], df[df.llm == llm]["correct"], num_bins=10
        )
        scaled_ece = cal.get_ece(
            df[df.llm == llm]["scaled_confidence"],
            df[df.llm == llm]["correct"],
            num_bins=10,
        )
        print(f"LLM {llm}: Unscaled ECE = {unscaled_ece}; Scaled ECE = {scaled_ece}")


@plot_method
def hallucination_by_importance():
    query = list(
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            (pw.fn.SUM(Case.importance) / pw.fn.COUNT()).alias("mean_importance"),
            TaskRun.llm.alias("llm"),
        )
        .join(Case)
        .group_by(pw.fn.ROUND(Case.importance * 50), TaskRun.llm)
        .where(
            (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.court_level == "scotus")
        )
        .dicts()
    )
    df = pd.DataFrame.from_records(query)
    df["mean_importance"] = df["mean_importance"] * 100
    df = df[df["mean_importance"] > 10]
    sns.lineplot(
        data=df,
        x="mean_importance",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        style="llm",
        dashes=False,
        legend=True,
        alpha=0.8,
    )
    if False:
        sns.scatterplot(
            data=df,
            x="mean_importance",
            y="hallucination_rate",
            hue="llm",
            hue_order=[llm.value for llm in Llm],
            style="llm",
        )
    # add an arrow saying "more prominent" is towards 100
    plt.annotate(
        "More Prominent",
        xy=(25, 0.90),
        xytext=(75, 0.90),
        arrowprops=dict(arrowstyle="<|-", color="black"),
        fontsize=6,
        # center text on arrow
        ha="center",
        va="center",
    )

    plt.xlabel("Case Prominence")
    plt.ylabel("Mean Hallucination Rate")
    plt.ylim(0, 1)
    plt.gca().legend(fontsize=6, frameon=False).set_title("")
    plt.gcf().tight_layout()


@plot_method
def hallucination_by_court_level():
    hallucination_rates = (
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            TaskRun.llm.alias("llm"),
            TaskRun.court_level.alias("court_level"),
        )
        .where(
            ~(TaskRun.task % "fake*")
            & (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
        )
        .group_by(TaskRun.llm, TaskRun.court_level)
    )
    df = pd.DataFrame.from_records(hallucination_rates.dicts())
    df.loc[df["court_level"] == "scotus", "court_level"] = "0_scotus"
    df.loc[df["court_level"] == "coa", "court_level"] = "1_coa"
    df.loc[df["court_level"] == "usdc", "court_level"] = "2_usdc"
    df = df.sort_values("court_level")
    df.loc[df["court_level"] == "0_scotus", "court_level"] = "SCOTUS"
    df.loc[df["court_level"] == "1_coa", "court_level"] = "USCOA"
    df.loc[df["court_level"] == "2_usdc", "court_level"] = "USDC"
    sns.barplot(
        data=df,
        x="court_level",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
    )
    plt.ylabel("Mean Hallucination Rate")
    plt.xlabel("")
    # set in upper left
    # shrink space between columns
    plt.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper left").set_title("")
    plt.ylim(0, 1)


@plot_method
def false_premise_hallucination_rate():
    hallucination_rates = (
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            TaskRun.llm.alias("llm"),
        )
        .where((TaskRun.task % "fake*") & (TaskRun.task != "fake_case_existence"))
        .group_by(TaskRun.llm)
    )
    df = pd.DataFrame.from_records(hallucination_rates.dicts())
    df["llm"] = pd.Categorical(
        df["llm"], categories=[llm.value for llm in Llm], ordered=True
    )
    for i, llm in enumerate([llm.value for llm in Llm]):
        row = df[df["llm"] == llm].iloc[0]
        plt.text(
            i,
            row["hallucination_rate"] + 0.025,
            f"{row['hallucination_rate']:.2f}",
            ha="center",
        )
    sns.barplot(
        data=df,
        x="llm",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
    )
    # nolegend
    plt.legend().remove()
    plt.ylabel("Mean Hallucination Rate")
    plt.xlabel("")
    plt.ylim(0, 1)


@plot_method
def hallucination_by_complexity():
    complexity = pw.Case(
        None,
        (
            (TaskRun.task.in_(LOW_COMPLEXITY_TASKS), "Low"),
            (TaskRun.task.in_(MEDIUM_COMPLEXITY_TASKS), "Medium"),
            (TaskRun.task.in_(HIGH_COMPLEXITY_TASKS), "High"),
        ),
        "Other Complexity",
    )

    tasks = (
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            TaskRun.llm.alias("llm"),
            complexity.alias("complexity"),
        )
        .where(
            (
                (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
                & TaskRun.task.in_(LOW_COMPLEXITY_TASKS + MEDIUM_COMPLEXITY_TASKS)
            )
            | (TaskRun.task.in_(HIGH_COMPLEXITY_TASKS))
        )
        .group_by(TaskRun.llm, complexity)
    )
    df = pd.DataFrame.from_records(tasks.dicts())
    df.sort_values(
        "complexity",
        inplace=True,
        key=lambda x: x.map({"Low": 0, "Medium": 1, "High": 2}),
    )
    sns.barplot(
        data=df,
        x="complexity",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        legend=False,
    )
    # containers
    high_complexity_rectangles = [container[-1] for container in plt.gca().containers]
    top_centers = [
        (rect.xy[0] + rect.get_width() / 2, rect.xy[1] + rect.get_height())
        for rect in high_complexity_rectangles
    ]
    plt.errorbar(
        x=[top_center[0] for top_center in top_centers],
        y=[top_center[1] for top_center in top_centers],
        yerr=[0.02],
        lolims=[True],
        color="black",
        capsize=2,
        elinewidth=0.5,
        # don't draw the line
        fmt="none",
    )

    for llm, container in zip([llm.value for llm in Llm], plt.gca().containers):
        for rect in container:
            bc = (rect.xy[0] + rect.get_width() / 2, rect.xy[1])
            plt.text(
                bc[0] + 0.05,
                bc[1] - 0.02,
                llm,
                ha="right",
                fontsize=6,
                rotation=45,
                horizontalalignment="right",
                verticalalignment="top",
            )

    plt.ylabel("Mean Hallucination Rate")
    # move xlabel and ticks to the top
    plt.xlabel("Task Complexity")
    plt.gca().xaxis.set_label_position("top")
    plt.gca().xaxis.tick_top()
    plt.gca().tick_params(axis="x", which="both", length=0)
    plt.legend(
        fontsize=6,
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        loc="upper center",
    ).set_title("")
    plt.ylim(0, 1)
    plt.tight_layout()


@plot_method
def hallucination_by_year():
    query = list(
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            (pw.fn.SUM(Case.year) / pw.fn.COUNT(Case.year)).alias("year"),
            TaskRun.llm.alias("llm"),
        )
        .join(Case)
        .group_by(pw.fn.ROUND(Case.year / 5), TaskRun.llm)
        .where(
            (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.court_level == "scotus")
        )
        .dicts()
    )
    df = pd.DataFrame.from_records(query)
    df = df[df["year"] > 1800]
    sns.lineplot(
        data=df,
        x="year",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        style="llm",
        dashes=False,
        legend=True,
        alpha=0.8,
    )
    if False:
        sns.scatterplot(
            data=df,
            x="year",
            y="hallucination_rate",
            hue="llm",
            hue_order=[llm.value for llm in Llm],
            style="llm",
        )
    plt.xlabel("Case Year")
    plt.ylabel("Mean Hallucination Rate")
    plt.ylim(0, 1)
    plt.gca().legend(fontsize=6, frameon=False).set_title("")
    plt.gcf().tight_layout()


@plot_method
def overall_hallucination_rate():
    hallucination_rates = (
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            TaskRun.llm.alias("llm"),
        )
        .where(
            (~(TaskRun.task % "fake*"))
            & (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.correctness_labeler == None)
        )
        .group_by(TaskRun.llm)
    )
    df = pd.DataFrame.from_records(hallucination_rates.dicts())
    df["llm"] = pd.Categorical(
        df["llm"], categories=[llm.value for llm in Llm], ordered=True
    )
    # print numbers at top of plot
    for i, llm in enumerate([llm.value for llm in Llm]):
        row = df[df["llm"] == llm].iloc[0]
        plt.text(
            i,
            row["hallucination_rate"] + 0.025,
            f"{row['hallucination_rate']:.2f}",
            ha="center",
        )
    sns.barplot(
        data=df,
        x="llm",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
    )
    plt.ylabel("Mean Hallucination Rate")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.tight_layout()


@plot_method
def author_inductive_bias():
    answer = lambda x: fn.json_extract(x, "$.answer")
    no_titles = lambda x: fn.replace(
        fn.replace(fn.replace(answer(x), "II", ""), "Jr.", ""), "Justice", ""
    )
    no_first_name = lambda x: fn.substr(no_titles(x), fn.instr(no_titles(x), " ") + 1)
    no_middle_name = lambda x: fn.substr(
        no_first_name(x), fn.instr(no_first_name(x), " ") + 1
    )
    last_name = lambda x: fn.trim(fn.coalesce(no_middle_name(x), no_first_name(x)))

    # get first initial of name, add period to end
    first_initial = lambda x: fn.substr(no_titles(x), 1, 1).concat(". ")

    # first initial, last name if marshall, else last name
    name = lambda x: pw.Case(
        None,
        (
            (last_name(x) == "Marshall", first_initial(x).concat(last_name(x))),
            (last_name(x) != "Marshall", last_name(x)),
        ),
        "name",
    )

    total_number_by_llm_cte = (
        TaskRun.select(
            TaskRun.llm.alias("llm"),
            (pw.fn.COUNT("*")).alias("count"),
        )
        .where(
            (TaskRun.task == Task.AUTHOR.value)
            & (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.court_level == CourtLevel.SCOTUS.value)
        )
        .group_by(TaskRun.llm)
        .cte("total_number_by_llm")
    )

    tr = TaskRun.alias()
    true_count_cte = (
        tr.select(
            tr.llm.alias("llm"),
            name(tr.true_answer).alias("justice"),
            (pw.fn.COUNT("*")).alias("count"),
        )
        .where(
            (tr.task == Task.AUTHOR.value)
            & (tr.prompt_style == PromptStyle.FEW_SHOT.value)
            & (tr.court_level == CourtLevel.SCOTUS.value)
        )
        .group_by(name(tr.true_answer), tr.llm)
        .cte("true_count")
    )

    llm_responses = (
        TaskRun.select(
            name(TaskRun.llm_greedy_answer).alias("justice"),
            TaskRun.llm.alias("llm"),
            (pw.fn.COUNT(TaskRun.llm_greedy_answer)).alias("count"),
            fn.coalesce(true_count_cte.c.count, 0).alias("true_count"),
            (
                pw.fn.COUNT(TaskRun.llm_greedy_answer) / total_number_by_llm_cte.c.count
            ).alias("response_proportion"),
            (true_count_cte.c.count / total_number_by_llm_cte.c.count).alias(
                "true_proportion"
            ),
        )
        .join(
            true_count_cte,
            on=(
                (true_count_cte.c.justice == name(TaskRun.llm_greedy_answer))
                & (true_count_cte.c.llm == TaskRun.llm)
            ),
            join_type=pw.JOIN.LEFT_OUTER,
        )
        .join(
            total_number_by_llm_cte,
            on=(total_number_by_llm_cte.c.llm == TaskRun.llm),
        )
        .where(
            (TaskRun.task == Task.AUTHOR.value)
            & (TaskRun.prompt_style == PromptStyle.FEW_SHOT.value)
            & (TaskRun.court_level == CourtLevel.SCOTUS.value)
        )
        .group_by(name(TaskRun.llm_greedy_answer), TaskRun.llm)
        .with_cte(total_number_by_llm_cte, true_count_cte)
        .dicts()
    )
    df = pd.DataFrame.from_records(llm_responses)
    df["llm"] = pd.Categorical(df["llm"], categories=[llm.value for llm in Llm])
    # make scatter transparent
    sns.scatterplot(
        data=df,
        y="count",
        x="true_count",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
        alpha=0.6,
        s=6,
    )
    # label the largest points
    drawn_points = []
    # sort df to go from largest row["count"] to smallest
    df = df.sort_values("count", ascending=False)
    for i, row in df.iterrows():
        if ((row["count"] > 380) or (row["true_count"] > 300)) or (
            (row["count"] > 300) and (row["true_count"] > 150)
        ):
            # if there is a point just below, draw the label above
            va = "top"
            if any(
                [
                    (abs(row["true_count"] - drawn_point[0]) < 30)
                    and (row["count"] < drawn_point[1])
                    and ((drawn_point[1] - row["count"]) < 20)
                    for drawn_point in drawn_points
                ]
            ):
                va = "bottom"
            offset = 5 if va == "bottom" else -5
            if row["justice"] == "Brennan":
                offset = 0
            plt.text(
                row["true_count"] + 5,
                row["count"] + offset,
                row["justice"],
                fontsize=5,
                ha="left",
                va=va,
            )
            drawn_points.append((row["true_count"], row["count"]))
    # set alpha=0.5 for all points
    # scale 0 to 1
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    # y=x line
    plt.plot([0, 1300], [0, 1300], linestyle="--", color="lightgrey")
    plt.ylabel("Number of LLM Responses")
    plt.xlabel("True Number in Dataset")
    # draw rectangles with the color instead of markers
    import matplotlib.patches as mpatches

    plt.legend(
        fontsize=6,
        frameon=False,
        handles=[
            mpatches.Patch(color=sns.color_palette()[i], alpha=0.6, label=llm.value)
            for i, llm in enumerate(Llm)
        ],
        loc="lower right",
    ).set_title("")
    # increase size of plot
    plt.gcf().tight_layout()


@plot_method
def zero_shot_overall():
    hallucination_rates = (
        TaskRun.select(
            HALLUCINATION_RATE.alias("hallucination_rate"),
            TaskRun.llm.alias("llm"),
        )
        .where(
            (~(TaskRun.task % "fake*"))
            & (TaskRun.prompt_style == PromptStyle.ZERO_SHOT.value)
            & (TaskRun.correctness_labeler == None)
        )
        .group_by(TaskRun.llm)
    )
    df = pd.DataFrame.from_records(hallucination_rates.dicts())
    df["llm"] = pd.Categorical(
        df["llm"], categories=[llm.value for llm in Llm], ordered=True
    )
    # print numbers at top of plot
    for i, llm in enumerate([llm.value for llm in Llm]):
        row = df[df["llm"] == llm].iloc[0]
        plt.text(
            i,
            row["hallucination_rate"] + 0.025,
            f"{row['hallucination_rate']:.2f}",
            ha="center",
        )
    sns.barplot(
        data=df,
        x="llm",
        y="hallucination_rate",
        hue="llm",
        hue_order=[llm.value for llm in Llm],
    )
    plt.ylabel("Mean Hallucination Rate, Zero-Shot")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.tight_layout()


def task_counts():
    query = (
        TaskRun.select(
            TaskRun.task.alias("task"),
            TaskRun.llm.alias("llm"),
            TaskRun.prompt_style.alias("prompt_style"),
            TaskRun.court_level.alias("court_level"),
            (pw.fn.COUNT(TaskRun.task)).alias("count"),
        )
        .where((TaskRun.correctness_labeler == None))
        .group_by(TaskRun.task, TaskRun.llm, TaskRun.prompt_style, TaskRun.court_level)
        .dicts()
    )
    df = pd.DataFrame.from_records(data=query)
    df.to_csv("task_counts.csv")


if __name__ == "__main__":
    for plot in tqdm(plot_methods):
        plot()
        plt.gcf().set_size_inches(3, 2.5)
        plt.savefig(
            os.path.join(FIGURES_SAVE_PATH, f"{FORMAT}/{plot.__name__}.{FORMAT}")
        )
        plt.savefig(
            os.path.join(FIGURES_SAVE_PATH, f"pngs/{plot.__name__}.png"),
            transparent=True,
        )
        plt.clf()
