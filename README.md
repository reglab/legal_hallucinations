# Large Legal Fictions

[![arXiv](https://img.shields.io/badge/arXiv-2401.01301-b31b1b.svg)](https://arxiv.org/abs/2401.01301)

This code accompanies the paper [Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models](https://arxiv.org/abs/2401.01301), authored by Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E. Ho.

Please cite our preprint as follows:

```
@misc{dahl2024largelegalfictions,
    title = {Large {{Legal Fictions}}: {{Profiling Legal Hallucinations}} in {{Large Language Models}}},
    author = {Matthew Dahl and Varun Magesh and Mirac Suzgun and Daniel E. Ho},
    year = {2024},
    eprint = {2401.01301},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

Please contact matthew.dahl@yale.edu and vim@law.stanford.edu with questions regarding the code.

# Replication procedure

## 1. Install requirements

First, install Python 3.11 or higher. Then, install the necessary Python packages:

```
pip install -r requirements.txt
```

We recommend installing these into a local Python virtual environment, and then performing all the remaining steps from within that environment.

If you want to run Llama locally (requires A100 or comparable GPU), you will also need to install Manifest:

```
pip install manifest-ml
pip install manifest-ml[api]
```

## 2. Set environment variables

Set the `BASE_DIRECTORY` variable appropriately in `settings.py`.

## 3. Set your API keys

API keys are stored in `reglab_secrets.py`; create your own secrets file with your own keys as appropriate. Required entries are:

```
OPENAI_API_KEY: str = 'YOUR_KEY_HERE'
PALM_API_KEY1: str = 'YOUR_KEY_HERE'
```

## 4. Prepare input data

We create our data set by sampling cases at random from the U.S., Federal, and Federal Supplement reporters (conditional on strata) and then by merging those cases with additional tabular metadata. To replicate this sampling and merging process from scratch, first populate the `data/sources/` directory with the data source files stored on [Hugging Face](https://huggingface.co/datasets/reglab/legal_hallucinations_paper_data/tree/main/sources). See Online Appendix Table 1 for a description of how each of these sources is used.

Then, to generate the samples used in this paper, run:

```
python data.py
```

In the alternative, to bypass the generation process entirely, simply populate the `data/samples/` directory with the data sample files stored on [Hugging Face](https://huggingface.co/datasets/reglab/legal_hallucinations_paper_data/tree/main/samples).

## 5. Run query tasks

To perform the main hallucination query tasks, run the following scripts:

```
python tasks_scotus.py
python tasks_coa.py
python tasks_usdc.py
python tasks_contrafactual.py
python tasks_zero_resource.py
```

Note that because these scripts are API-bound, a full evaluation of the pipeline will take several days to complete. At the top of each `tasks_` script is a `CURRENT_API` flag that must be set to the API to use for the run. You can configure this manually or pass it as a command-line argument using the `--api` flag. Valid options include:

- `OpenAIChatGpt4` -> GPT 4
- `OpenAIChat` -> GPT 3.5
- `GooglePaLMCompletion` -> PaLM 2
- `LlamaChat` -> Llama 2 (if running locally using `manifest`)
- `TogetherAiLlamaChat` -> Llama 2 (if using `Together AI`)

If using `LlamaChat`, the local Manifest server for the model must first be started:

```
python -m manifest.api.app --model_type huggingface --model_name_or_path <PATH_TO_MODEL_DIRECTORY> --model_generation_type text-generation --device 0
```

## 6. Generate tables

To generate the tables, run:

```
Rscript tables.R
```

(Note that the R packages imported at the top of this script must be installed. You can also run this script from within Rstudio or another R environment.)


## 7. Generate figures

To generate the figures, first run these load functions. You only need to do this once.

```
$ python
>>> from plot import *
>>> Case.load(); TaskRun.load();
```

After that, to generate all the axis plots used in the paper, run:

```
python plot.py
```

To generate the geographic plots, run:

```
Rscript plot_maps.R
```

(Note that the R packages imported at the top of this script must be installed. You can also run this script from within Rstudio or another R environment.)

# Implementation details

As noted above, the `tasks_*.py` scripts serve as the user-facing interface. Within each of these scripts, individual tasks are instantiated as `Task` objects, which accept a list of `Query` objects representing the queries to be run for each task (normally 5000). Each task is then executed by calling its `.do()` method. The individual queries can be parallelized; set the `NUM_THREADS` flag in `settings.py` to the number of threads to use. Tasks are saved by calling their `.save()` method.

More implementation details are as follows:

- `api.py`: Abstractions for interfacing with different LLM backends. Note that some of these implementions are not currently used. As noted above, supported API backends are `[OpenAIChatGpt4, OpenAIChat, GooglePaLMCompletion, LlamaChat, TogetherAiLlamaChat]`.
- `correctness_checks.py`: Functions for cleaning LLM responses and determining whether they are hallucinations or not.
- `models.py`: Classes for handling different parts of the pipeline. (Note: These are not models in the LLM sense; only in the OOP sense.)
- `utils.py`: General utility functions and type definitions.

Our code is fully-typed to improve clarity and minimize errors. Run `mypy .  --ignore-missing-imports` to check for problems.
