#################################
# Setup
#################################

# Libs
library(knitr)
library(kableExtra)
library(stringr)
library(dplyr)
library(stats)
library(reticulate)
cal <- import("calibration")
np <- import("numpy")
settings <- import("settings")

# Flags
set.seed(47)
BOOTSTRAP_ITERATIONS = 100
RESULTS_DIR = str_glue("{settings$BASE_DIRECTORY}/results")
TABLES_DIR = str_glue("{settings$BASE_DIRECTORY}/results/tables")

# Helper function for calculating binomial standard errors
get_binomial_se <- function (p, n) {
  sqrt(p * (1 - p) / n)
}

# Helper function to retrieve hallucination rates
get_hr <- function (court, task, api, temp, balance = FALSE) {
  # Load model results
  setwd(str_glue("{RESULTS_DIR}/tasks/{court}"))
  result <- read.csv(str_glue("{task}_{api}_results_temp={temp}.csv"))

  # Set hallucination cutoff
  result$hallucination <- ifelse((result$correctness <= 72) & (result$correctness != -99), 1, 0)

  if (balance == TRUE) {
    # Calculate hallucination rate point estimate
    stopifnot(length(names(table(result$true_answer))) == 2)  # Ensure that there are indeed only two classes

    class1 <- result %>% filter(true_answer == names(table(result$true_answer))[1])
    class2 <- result %>% filter(true_answer == names(table(result$true_answer))[2])

    class1_hr <- mean(class1$hallucination)
    class2_hr <- mean(class2$hallucination)
    hr <- (1 / 2) * (class1_hr + class2_hr)

    # Calculate hallucination rate SE
    class1_var <- (class1_hr * (1 - class1_hr)) / nrow(class1)
    class2_var <- (class2_hr * (1 - class2_hr)) / nrow(class2)
    total_var <- (1 / 4) * (class1_var + class2_var)  # the 1/2 constant becomes squared (properties of variance)
    hr_se <- sqrt(total_var)
  } else {
    # Calculate hallucination rate point estimate
    hr <- mean(result$hallucination)

    # Calculate hallucination rate SE
    hr_se <- get_binomial_se(hr, nrow(result))
  }

  # Return a string in EST (SE) form
  return(str_glue("{sprintf('%.3f', hr)}\n({sprintf('%.3f', hr_se)})"))
}

# Helper function to retrieve abstention rates
get_no_answer_rate <- function (court, task, api, temp) {
  # Load model results
  setwd(str_glue("{RESULTS_DIR}/tasks/{court}"))
  result <- read.csv(str_glue("{task}_{api}_results_temp={temp}.csv"))

  # Calculate abstention rate point estimate
  rate <- round(nrow(filter(result, correctness == -99)) / nrow(result), digits = 3)

  # Calculate abstention rate SE
  rate_se <- get_binomial_se(rate, nrow(result))

  # Return a string in EST (SE) form
  return(str_glue("{sprintf('%.3f', rate)}\n({sprintf('%.3f', rate_se)})"))
}

# Helper function to calculate raw and temperature-scaled ECE
get_ece <- function (court, task, api, temp, rescale = FALSE) {
  # Load model results
  setwd(str_glue("{RESULTS_DIR}/tasks/{court}"))
  result <- read.csv(str_glue("{task}_{api}_results_temp={temp}.csv"))

  # Filter out decline-to-answer responses
  result <- result %>% filter(correctness != -99)

  # Set hallucination cutoff
  result$hallucination <- ifelse(result$correctness <= 72, 1, 0)

  # Prepare variables to pass to calibration package
  result$correct <- 1 - result$hallucination
  result$confidence <- result$confidence / 100

  # Do temperature scaling if requested
  if (rescale == TRUE & length(unique(result$correct)) == 2) {
    calibrator <- cal$PlattCalibrator(nrow(result), num_bins = as.integer(10))
    calibrator$train_calibration(np$array(result$confidence), as.integer(result$correct))
    result$confidence <- calibrator$calibrate(result$confidence)
  }

  # Calculate ECE point estimate
  ece <- cal$get_ece(result$confidence, as.integer(result$correct), num_bins = as.integer(10))

  # Calculate standard error via bootstrapping
  ece_estimates <- double(BOOTSTRAP_ITERATIONS)
  for (i in 1:BOOTSTRAP_ITERATIONS) {
    subsample_indicies <- sample(1:nrow(result), size = nrow(result), replace = TRUE)
    subsample_df <- result[subsample_indicies,]
    ece_estimates[i] <- cal$get_ece(subsample_df$confidence, as.integer(subsample_df$correct), num_bins = as.integer(10))
  }

  # Return a string in EST (SE) form
  return(str_glue("{sprintf('%.3f', ece)}\n({sprintf('%.3f', sd(ece_estimates))})"))
}

# Common df column names
df_column_names <- c("Task", "Prompt", "GPT 4", "GPT 3.5", "PaLM 2", "Llama 2", "GPT 4", "GPT 3.5", "PaLM 2", "Llama 2", "GPT 4", "GPT 3.5", "PaLM 2", "Llama 2")
scotus <- "SCOTUS\n(1794-2015; n=5000)"
coa <- "USCOA\n(1895-2019; n=5000)"
usdc <- "USDC\n(1932-2019; n=5000)"


#################################
# Table 1: Hallucination rates (low complexity tasks)
#################################

# Create df
tasks <- c("Existence", "Existence", "Court", "Court", "Citation", "Citation", "Author", "Author")
prompts <- rep(c("Zero-shot", "Few-shot"), 4)
df <- data.frame(
  Task = tasks,
  Prompt = prompts,
  ChatGPT4_SCOTUS = c(
    get_hr("scotus", "case_existence", "OpenAIChatGpt4", 1),
    get_hr("scotus", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_hr("scotus", "court_id", "OpenAIChatGpt4", 1),
    get_hr("scotus", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_hr("scotus", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_hr("scotus", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_hr("scotus", "majority_author", "OpenAIChatGpt4", 1),
    get_hr("scotus", "majority_author_few_shot", "OpenAIChatGpt4", 1)),
  ChatGPT_SCOTUS = c(
    get_hr("scotus", "case_existence", "OpenAIChat", 1),
    get_hr("scotus", "case_existence_few_shot", "OpenAIChat", 1),
    get_hr("scotus", "court_id", "OpenAIChat", 1),
    get_hr("scotus", "court_id_few_shot", "OpenAIChat", 1),
    get_hr("scotus", "citation_retrieval", "OpenAIChat", 1),
    get_hr("scotus", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_hr("scotus", "majority_author", "OpenAIChat", 1),
    get_hr("scotus", "majority_author_few_shot", "OpenAIChat", 1)),
  PaLM_SCOTUS = c(
    get_hr("scotus", "case_existence", "GooglePaLMCompletion", 1),
    get_hr("scotus", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_hr("scotus", "court_id", "GooglePaLMCompletion", 1),
    get_hr("scotus", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_hr("scotus", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_hr("scotus", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_hr("scotus", "majority_author", "GooglePaLMCompletion", 1),
    get_hr("scotus", "majority_author_few_shot", "GooglePaLMCompletion", 1)),
  Llama_SCOTUS = c(
    get_hr("scotus", "case_existence", "LlamaChat", 1),
    get_hr("scotus", "case_existence_few_shot", "LlamaChat", 1),
    get_hr("scotus", "court_id", "LlamaChat", 1),
    get_hr("scotus", "court_id_few_shot", "LlamaChat", 1),
    get_hr("scotus", "citation_retrieval", "LlamaChat", 1),
    get_hr("scotus", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_hr("scotus", "majority_author", "LlamaChat", 1),
    get_hr("scotus", "majority_author_few_shot", "LlamaChat", 1)),
  ChatGPT4_COA = c(
    get_hr("coa", "case_existence", "OpenAIChatGpt4", 1),
    get_hr("coa", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_hr("coa", "court_id", "OpenAIChatGpt4", 1),
    get_hr("coa", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_hr("coa", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_hr("coa", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_hr("coa", "majority_author", "OpenAIChatGpt4", 1),
    get_hr("coa", "majority_author_few_shot", "OpenAIChatGpt4", 1)),
  ChatGPT_COA = c(
    get_hr("coa", "case_existence", "OpenAIChat", 1),
    get_hr("coa", "case_existence_few_shot", "OpenAIChat", 1),
    get_hr("coa", "court_id", "OpenAIChat", 1),
    get_hr("coa", "court_id_few_shot", "OpenAIChat", 1),
    get_hr("coa", "citation_retrieval", "OpenAIChat", 1),
    get_hr("coa", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_hr("coa", "majority_author", "OpenAIChat", 1),
    get_hr("coa", "majority_author_few_shot", "OpenAIChat", 1)),
  PaLM_COA = c(
    get_hr("coa", "case_existence", "GooglePaLMCompletion", 1),
    get_hr("coa", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_hr("coa", "court_id", "GooglePaLMCompletion", 1),
    get_hr("coa", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_hr("coa", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_hr("coa", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_hr("coa", "majority_author", "GooglePaLMCompletion", 1),
    get_hr("coa", "majority_author_few_shot", "GooglePaLMCompletion", 1)),
  Llama_COA = c(
    get_hr("coa", "case_existence", "LlamaChat", 1),
    get_hr("coa", "case_existence_few_shot", "LlamaChat", 1),
    get_hr("coa", "court_id", "LlamaChat", 1),
    get_hr("coa", "court_id_few_shot", "LlamaChat", 1),
    get_hr("coa", "citation_retrieval", "LlamaChat", 1),
    get_hr("coa", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_hr("coa", "majority_author", "LlamaChat", 1),
    get_hr("coa", "majority_author_few_shot", "LlamaChat", 1)),
  ChatGPT4_USDC = c(
    get_hr("usdc", "case_existence", "OpenAIChatGpt4", 1),
    get_hr("usdc", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_hr("usdc", "court_id", "OpenAIChatGpt4", 1),
    get_hr("usdc", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_hr("usdc", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_hr("usdc", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_hr("usdc", "majority_author", "OpenAIChatGpt4", 1),
    get_hr("usdc", "majority_author_few_shot", "OpenAIChatGpt4", 1)),
  ChatGPT_USDC = c(
    get_hr("usdc", "case_existence", "OpenAIChat", 1),
    get_hr("usdc", "case_existence_few_shot", "OpenAIChat", 1),
    get_hr("usdc", "court_id", "OpenAIChat", 1),
    get_hr("usdc", "court_id_few_shot", "OpenAIChat", 1),
    get_hr("usdc", "citation_retrieval", "OpenAIChat", 1),
    get_hr("usdc", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_hr("usdc", "majority_author", "OpenAIChat", 1),
    get_hr("usdc", "majority_author_few_shot", "OpenAIChat", 1)),
  PaLM_USDC = c(
    get_hr("usdc", "case_existence", "GooglePaLMCompletion", 1),
    get_hr("usdc", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_hr("usdc", "court_id", "GooglePaLMCompletion", 1),
    get_hr("usdc", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_hr("usdc", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_hr("usdc", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_hr("usdc", "majority_author", "GooglePaLMCompletion", 1),
    get_hr("usdc", "majority_author_few_shot", "GooglePaLMCompletion", 1)),
  Llama_USDC = c(
    get_hr("usdc", "case_existence", "LlamaChat", 1),
    get_hr("usdc", "case_existence_few_shot", "LlamaChat", 1),
    get_hr("usdc", "court_id", "LlamaChat", 1),
    get_hr("usdc", "court_id_few_shot", "LlamaChat", 1),
    get_hr("usdc", "citation_retrieval", "LlamaChat", 1),
    get_hr("usdc", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_hr("usdc", "majority_author", "LlamaChat", 1),
    get_hr("usdc", "majority_author_few_shot", "LlamaChat", 1))
)
df <- df %>% mutate_all(linebreak, align = c("c"))
names(df) <- df_column_names

# Create a LaTeX table
t <- df %>%
  kable(format = "latex", booktabs = TRUE, caption = "Hallucination rates across levels of the federal judiciary (low complexity tasks)", escape = FALSE, linesep = "", table.env = "table*", label = "CategoryOneTable") %>%
  collapse_rows(columns = 1) %>%
  add_header_above(c(" " = 2, setNames(4, scotus), setNames(4, coa), setNames(4, usdc))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    footnote_order = c("general"),
    general = "\\\\\\\\ \\\\textit{Note:} Table reports estimated hallucination rates. Standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
t <- gsub("\\multirow{-2}{*}", "\\multirow{-2}{*}[1em]", t, fixed = TRUE)
f <- str_glue("{TABLES_DIR}/CategoryOneTable.tex")
cat(paste(t, collapse = "\n"), "\n", file = f)
writeLines(str_glue("Saved table {f}"))

#################################
# Table 2: Hallucination rates (moderate complexity tasks)
#################################

# Create df
tasks2 <- c("Disposition", "Disposition", "Quotation", "Quotation", "Authority", "Authority", paste0("Overruling year", footnote_marker_alphabet(1, format = "latex", double_escape = FALSE)), paste0("Overruling year", footnote_marker_alphabet(1, format = "latex", double_escape = FALSE)))
prompts2 <- rep(c("Zero-shot", "Few-shot"), 4)
df2 <- data.frame(
  Task = tasks2,
  Prompt = prompts2,
  ChatGPT4_SCOTUS = c(
    get_hr("scotus", "affirm_reverse", "OpenAIChatGpt4", 1, balance = TRUE),
    get_hr("scotus", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1, balance = TRUE),
    get_hr("scotus", "quotation", "OpenAIChatGpt4", -99),
    get_hr("scotus", "quotation_few_shot", "OpenAIChatGpt4", -99),
    get_hr("scotus", "cited_precedent", "OpenAIChatGpt4", -99),
    get_hr("scotus", "cited_precedent_few_shot", "OpenAIChatGpt4", -99),
    get_hr("scotus", "year_overruled", "OpenAIChatGpt4", 1),
    get_hr("scotus", "year_overruled_few_shot", "OpenAIChatGpt4", 1)),
  ChatGPT_SCOTUS = c(
    get_hr("scotus", "affirm_reverse", "OpenAIChat", 1, balance = TRUE),
    get_hr("scotus", "affirm_reverse_few_shot", "OpenAIChat", 1, balance = TRUE),
    get_hr("scotus", "quotation", "OpenAIChat", -99),
    get_hr("scotus", "quotation_few_shot", "OpenAIChat", -99),
    get_hr("scotus", "cited_precedent", "OpenAIChat", -99),
    get_hr("scotus", "cited_precedent_few_shot", "OpenAIChat", -99),
    get_hr("scotus", "year_overruled", "OpenAIChat", 1),
    get_hr("scotus", "year_overruled_few_shot", "OpenAIChat", 1)),
  PaLM_SCOTUS = c(
    get_hr("scotus", "affirm_reverse", "GooglePaLMCompletion", 1, balance = TRUE),
    get_hr("scotus", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1, balance = TRUE),
    get_hr("scotus", "quotation", "GooglePaLMCompletion", -99),
    get_hr("scotus", "quotation_few_shot", "GooglePaLMCompletion", -99),
    get_hr("scotus", "cited_precedent", "GooglePaLMCompletion", -99),
    get_hr("scotus", "cited_precedent_few_shot", "GooglePaLMCompletion", -99),
    get_hr("scotus", "year_overruled", "GooglePaLMCompletion", 1),
    get_hr("scotus", "year_overruled_few_shot", "GooglePaLMCompletion", 1)),
  Llama_SCOTUS = c(
    get_hr("scotus", "affirm_reverse", "LlamaChat", 1, balance = TRUE),
    get_hr("scotus", "affirm_reverse_few_shot", "LlamaChat", 1, balance = TRUE),
    get_hr("scotus", "quotation", "LlamaChat", -99),
    get_hr("scotus", "quotation_few_shot", "LlamaChat", -99),
    get_hr("scotus", "cited_precedent", "LlamaChat", -99),
    get_hr("scotus", "cited_precedent_few_shot", "LlamaChat", -99),
    get_hr("scotus", "year_overruled", "LlamaChat", 1),
    get_hr("scotus", "year_overruled_few_shot", "LlamaChat", 1)),
  ChatGPT4_COA = c(
    get_hr("coa", "affirm_reverse", "OpenAIChatGpt4", 1, balance = TRUE),
    get_hr("coa", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1, balance = TRUE),
    get_hr("coa", "quotation", "OpenAIChatGpt4", -99),
    get_hr("coa", "quotation_few_shot", "OpenAIChatGpt4", -99),
    get_hr("coa", "cited_precedent", "OpenAIChatGpt4", -99),
    get_hr("coa", "cited_precedent_few_shot", "OpenAIChatGpt4", -99),
    "-",
    "-"),
  ChatGPT_COA = c(
    get_hr("coa", "affirm_reverse", "OpenAIChat", 1, balance = TRUE),
    get_hr("coa", "affirm_reverse_few_shot", "OpenAIChat", 1, balance = TRUE),
    get_hr("coa", "quotation", "OpenAIChat", -99),
    get_hr("coa", "quotation_few_shot", "OpenAIChat", -99),
    get_hr("coa", "cited_precedent", "OpenAIChat", -99),
    get_hr("coa", "cited_precedent_few_shot", "OpenAIChat", -99),
    "-",
    "-"),
  PaLM_COA = c(
    get_hr("coa", "affirm_reverse", "GooglePaLMCompletion", 1, balance = TRUE),
    get_hr("coa", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1, balance = TRUE),
    get_hr("coa", "quotation", "GooglePaLMCompletion", -99),
    get_hr("coa", "quotation_few_shot", "GooglePaLMCompletion", -99),
    get_hr("coa", "cited_precedent", "GooglePaLMCompletion", -99),
    get_hr("coa", "cited_precedent_few_shot", "GooglePaLMCompletion", -99),
    "-",
    "-"),
  Llama_COA = c(
    get_hr("coa", "affirm_reverse", "LlamaChat", 1, balance = TRUE),
    get_hr("coa", "affirm_reverse_few_shot", "LlamaChat", 1, balance = TRUE),
    get_hr("coa", "quotation", "LlamaChat", -99),
    get_hr("coa", "quotation_few_shot", "LlamaChat", -99),
    get_hr("coa", "cited_precedent", "LlamaChat", -99),
    get_hr("coa", "cited_precedent_few_shot", "LlamaChat", -99),
    "-",
    "-"),
  ChatGPT4_USDC = c(
    "-",
    "-",
    get_hr("usdc", "quotation", "OpenAIChatGpt4", -99),
    get_hr("usdc", "quotation_few_shot", "OpenAIChatGpt4", -99),
    get_hr("usdc", "cited_precedent", "OpenAIChatGpt4", -99),
    get_hr("usdc", "cited_precedent_few_shot", "OpenAIChatGpt4", -99),
    "-",
    "-"),
  ChatGPT_USDC = c(
    "-",
    "-",
    get_hr("usdc", "quotation", "OpenAIChat", -99),
    get_hr("usdc", "quotation_few_shot", "OpenAIChat", -99),
    get_hr("usdc", "cited_precedent", "OpenAIChat", -99),
    get_hr("usdc", "cited_precedent_few_shot", "OpenAIChat", -99),
    "-",
    "-"),
  PaLM_USDC = c(
    "-",
    "-",
    get_hr("usdc", "quotation", "GooglePaLMCompletion", -99),
    get_hr("usdc", "quotation_few_shot", "GooglePaLMCompletion", -99),
    get_hr("usdc", "cited_precedent", "GooglePaLMCompletion", -99),
    get_hr("usdc", "cited_precedent_few_shot", "GooglePaLMCompletion", -99),
    "-",
    "-"),
  Llama_USDC = c(
    "-",
    "-",
    get_hr("usdc", "quotation", "LlamaChat", -99),
    get_hr("usdc", "quotation_few_shot", "LlamaChat", -99),
    get_hr("usdc", "cited_precedent", "LlamaChat", -99),
    get_hr("usdc", "cited_precedent_few_shot", "LlamaChat", -99),
    "-",
    "-")
)
df2 <- df2 %>% mutate_all(linebreak, align = c("c"))
names(df2) <- df_column_names

# Create a LaTeX table
t2 <- df2 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Hallucination rates across levels of the federal judiciary (moderate complexity tasks)", escape = FALSE, linesep = "", table.env = "table*", label = "CategoryTwoTable") %>%
  collapse_rows(columns = 1) %>%
  add_header_above(c(" " = 2, setNames(4, scotus), setNames(4, coa), setNames(4, usdc))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    footnote_order = c("alphabet", "general"),
    alphabet = c("1810-2022 (n=279)"),
    general = "\\\\\\\\ \\\\textit{Note:} Table reports estimated hallucination rates. Standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
t2 <- gsub("\\multirow{-2}{*}", "\\multirow{-2}{*}[1em]", t2, fixed = TRUE)
f2 <- str_glue("{TABLES_DIR}/CategoryTwoTable.tex")
cat(paste(t2, collapse = "\n"), "\n", file = f2)
writeLines(str_glue("Saved table {f2}"))


#################################
# Table 3: Hallucination rates (high complexity tasks)
#################################
scotus <- "SCOTUS\n(1794-2015; n=100)"
coa <- "USCOA\n(1895-2019; n=100)"
usdc <- "USDC\n(1932-2019; n=100)"

# Create df
tasks3 <- c(paste0("Doctrinal agreement", footnote_marker_alphabet(1, format = "latex", double_escape = FALSE)), paste0("Doctrinal agreement", footnote_marker_alphabet(1, format = "latex", double_escape = FALSE)), "Factual background", "Procedural posture", "Subsequent history", "Core legal question", "Central holding")
prompts3 <- c("Zero-shot", "Few-shot", rep("Zero-shot", 5))
df3 <- data.frame(
  Task = tasks3,
  Prompt = prompts3,
  ChatGPT4_SCOTUS = c(
    get_hr("scotus", "doctrinal_agreement", "OpenAIChatGpt4", 1, balance = TRUE),
    get_hr("scotus", "doctrinal_agreement_few_shot", "OpenAIChatGpt4", 1, balance = TRUE),
    get_hr("scotus", "factual_background", "OpenAIChatGpt4", 1),
    get_hr("scotus", "posture", "OpenAIChatGpt4", 1),
    get_hr("scotus", "subsequent_history", "OpenAIChatGpt4", 1),
    get_hr("scotus", "core_legal_question", "OpenAIChatGpt4", 1),
    get_hr("scotus", "holding", "OpenAIChatGpt4", 1)),
  ChatGPT_SCOTUS = c(
    get_hr("scotus", "doctrinal_agreement", "OpenAIChat", 1, balance = TRUE),
    get_hr("scotus", "doctrinal_agreement_few_shot", "OpenAIChat", 1, balance = TRUE),
    get_hr("scotus", "factual_background", "OpenAIChat", 1),
    get_hr("scotus", "posture", "OpenAIChat", 1),
    get_hr("scotus", "subsequent_history", "OpenAIChat", 1),
    get_hr("scotus", "core_legal_question", "OpenAIChat", 1),
    get_hr("scotus", "holding", "OpenAIChat", 1)),
  PaLM_SCOTUS = c(
    get_hr("scotus", "doctrinal_agreement", "GooglePaLMCompletion", 1, balance = TRUE),
    get_hr("scotus", "doctrinal_agreement_few_shot", "GooglePaLMCompletion", 1, balance = TRUE),
    get_hr("scotus", "factual_background", "GooglePaLMCompletion", 1),
    get_hr("scotus", "posture", "GooglePaLMCompletion", 1),
    get_hr("scotus", "subsequent_history", "GooglePaLMCompletion", 1),
    get_hr("scotus", "core_legal_question", "GooglePaLMCompletion", 1),
    get_hr("scotus", "holding", "GooglePaLMCompletion", 1)),
  Llama_SCOTUS = c(
    get_hr("scotus", "doctrinal_agreement", "LlamaChat", 1, balance = TRUE),
    get_hr("scotus", "doctrinal_agreement_few_shot", "LlamaChat", 1, balance = TRUE),
    get_hr("scotus", "factual_background", "LlamaChat", 1),
    get_hr("scotus", "posture", "LlamaChat", 1),
    get_hr("scotus", "subsequent_history", "LlamaChat", 1),
    get_hr("scotus", "core_legal_question", "LlamaChat", 1),
    get_hr("scotus", "holding", "LlamaChat", 1)),
  ChatGPT4_COA = c(
    "-",
    "-",
    get_hr("coa", "factual_background", "OpenAIChatGpt4", 1),
    get_hr("coa", "posture", "OpenAIChatGpt4", 1),
    get_hr("coa", "subsequent_history", "OpenAIChatGpt4", 1),
    get_hr("coa", "core_legal_question", "OpenAIChatGpt4", 1),
    get_hr("coa", "holding", "OpenAIChatGpt4", 1)),
  ChatGPT_COA = c(
    "-",
    "-",
    get_hr("coa", "factual_background", "OpenAIChat", 1),
    get_hr("coa", "posture", "OpenAIChat", 1),
    get_hr("coa", "subsequent_history", "OpenAIChat", 1),
    get_hr("coa", "core_legal_question", "OpenAIChat", 1),
    get_hr("coa", "holding", "OpenAIChat", 1)),
  PaLM_COA = c(
    "-",
    "-",
    get_hr("coa", "factual_background", "GooglePaLMCompletion", 1),
    get_hr("coa", "posture", "GooglePaLMCompletion", 1),
    get_hr("coa", "subsequent_history", "GooglePaLMCompletion", 1),
    get_hr("coa", "core_legal_question", "GooglePaLMCompletion", 1),
    get_hr("coa", "holding", "GooglePaLMCompletion", 1)),
  Llama_COA = c(
    "-",
    "-",
    get_hr("coa", "factual_background", "LlamaChat", 1),
    get_hr("coa", "posture", "LlamaChat", 1),
    get_hr("coa", "subsequent_history", "LlamaChat", 1),
    get_hr("coa", "core_legal_question", "LlamaChat", 1),
    get_hr("coa", "holding", "LlamaChat", 1)),
  ChatGPT4_USDC = c(
    "-",
    "-",
    get_hr("usdc", "factual_background", "OpenAIChatGpt4", 1),
    get_hr("usdc", "posture", "OpenAIChatGpt4", 1),
    get_hr("usdc", "subsequent_history", "OpenAIChatGpt4", 1),
    get_hr("usdc", "core_legal_question", "OpenAIChatGpt4", 1),
    get_hr("usdc", "holding", "OpenAIChatGpt4", 1)),
  ChatGPT_USDC = c(
    "-",
    "-",
    get_hr("usdc", "factual_background", "OpenAIChat", 1),
    get_hr("usdc", "posture", "OpenAIChat", 1),
    get_hr("usdc", "subsequent_history", "OpenAIChat", 1),
    get_hr("usdc", "core_legal_question", "OpenAIChat", 1),
    get_hr("usdc", "holding", "OpenAIChat", 1)),
  PaLM_USDC = c(
    "-",
    "-",
    get_hr("usdc", "factual_background", "GooglePaLMCompletion", 1),
    get_hr("usdc", "posture", "GooglePaLMCompletion", 1),
    get_hr("usdc", "subsequent_history", "GooglePaLMCompletion", 1),
    get_hr("usdc", "core_legal_question", "GooglePaLMCompletion", 1),
    get_hr("usdc", "holding", "GooglePaLMCompletion", 1)),
  Llama_USDC = c(
    "-",
    "-",
    get_hr("usdc", "factual_background", "LlamaChat", 1),
    get_hr("usdc", "posture", "LlamaChat", 1),
    get_hr("usdc", "subsequent_history", "LlamaChat", 1),
    get_hr("usdc", "core_legal_question", "LlamaChat", 1),
    get_hr("usdc", "holding", "LlamaChat", 1))
)
df3 <- df3 %>% mutate_all(linebreak, align = c("c"))
names(df3) <- df_column_names

# Create a LaTeX table
t3 <- df3 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Hallucination rates across levels of the federal judiciary (high complexity tasks)", escape = FALSE, linesep = "", table.env = "table*", label = "CategoryThreeTable") %>%
  collapse_rows(columns = 1) %>%
  add_header_above(c(" " = 2, setNames(4, scotus), setNames(4, coa), setNames(4, usdc))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    footnote_order = c("alphabet", "general"),
    alphabet = c("1796-2005 (n=5000)"),
    general = "\\\\\\\\ \\\\textit{Note:} Table reports estimated hallucination rates. For all tasks except doctrinal agreement, this rate is only a lower bound on the true population rate. Standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
t3 <- gsub("\\multirow{-2}{*}", "\\multirow{-2}{*}[1em]", t3, fixed = TRUE)
f3 <- str_glue("{TABLES_DIR}/CategoryThreeTable.tex")
cat(paste(t3, collapse = "\n"), "\n", file = f3)
writeLines(str_glue("Saved table {f3}"))


#################################
# Table 4: Hallucination rates (contra-factual tasks)
#################################

# Create df
tasks4 <- c(
  "False dissent premise",
  "False overruling premise"
)
prompts4 <- rep("Zero-shot", 2)
df4 <- data.frame(
  Task = tasks4,
  Prompt = prompts4,
  ChatGPT4_SCOTUS = c(
    get_hr("scotus", "fake_dissent", "OpenAIChatGpt4", -99),
    get_hr("scotus", "fake_year_overruled", "OpenAIChatGpt4", 1)
  ),
  ChatGPT_SCOTUS = c(
    get_hr("scotus", "fake_dissent", "OpenAIChat", -99),
    get_hr("scotus", "fake_year_overruled", "OpenAIChat", 1)
  ),
  PaLM_SCOTUS = c(
    get_hr("scotus", "fake_dissent", "GooglePaLMCompletion", -99),
    get_hr("scotus", "fake_year_overruled", "GooglePaLMCompletion", 1)
  ),
  Llama_SCOTUS = c(
    get_hr("scotus", "fake_dissent", "LlamaChat", -99),
    get_hr("scotus", "fake_year_overruled", "LlamaChat", 1)
  ),
  ChatGPT4_COA = c(
    get_hr("coa", "fake_dissent", "OpenAIChatGpt4", -99),
    "-"
  ),
  ChatGPT_COA = c(
    get_hr("coa", "fake_dissent", "OpenAIChat", -99),
    "-"
  ),
  PaLM_COA = c(
    get_hr("coa", "fake_dissent", "GooglePaLMCompletion", -99),
    "-"
  ),
  Llama_COA = c(
    get_hr("coa", "fake_dissent", "LlamaChat", -99),
    "-"
  )
)
df4 <- df4 %>% mutate_all(linebreak, align = c("c"))
names(df4) <- c("Task", "Prompt", "GPT 4", "GPT 3.5", "PaLM 2", "Llama 2", "GPT 4", "GPT 3.5", "PaLM 2", "Llama 2")

# Create a LaTeX table
t4 <- df4 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Hallucination rates across levels of the federal judiciary (contra-factual tasks)", escape = FALSE, linesep = "", table.env = "table*", label = "ContraryToFactTable") %>%
  add_header_above(c(" " = 2, setNames(4, "SCOTUS\n(1794-2015; n=1000)"), setNames(4, "USCOA\n(1895-2019; n=1000)"))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    general = "\\\\\\\\ \\\\textit{Note:} Table reports estimated hallucination rates. Standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
f4 <- str_glue("{TABLES_DIR}/ContraryToFactTable.tex")
cat(paste(t4, collapse = "\n"), "\n", file = f4)
writeLines(str_glue("Saved table {f4}"))


#################################
# Table 5: ECE (unscaled)
#################################

# Create df
tasks5 <- c("Existence", "Existence", "Court", "Court", "Citation", "Citation", "Author", "Author", "Disposition", "Disposition", paste0("Overruling year", footnote_marker_alphabet(1, format = "latex", double_escape = FALSE)), paste0("Overruling year", footnote_marker_alphabet(1, format = "latex", double_escape = FALSE)), paste0("Doctrinal agreement", footnote_marker_alphabet(2, format = "latex", double_escape = FALSE)), paste0("Doctrinal agreement", footnote_marker_alphabet(2, format = "latex", double_escape = FALSE)))
prompts5 <- rep(c("Zero-shot", "Few-shot"), 7)
df5 <- data.frame(
  Task = tasks5,
  Prompt = prompts5,
  ChatGPT4_SCOTUS = c(
    get_ece("scotus", "case_existence", "OpenAIChatGpt4", 1),
    get_ece("scotus", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_ece("scotus", "court_id", "OpenAIChatGpt4", 1),
    get_ece("scotus", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_ece("scotus", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_ece("scotus", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_ece("scotus", "majority_author", "OpenAIChatGpt4", 1),
    get_ece("scotus", "majority_author_few_shot", "OpenAIChatGpt4", 1),
    get_ece("scotus", "affirm_reverse", "OpenAIChatGpt4", 1),
    get_ece("scotus", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1),
    get_ece("scotus", "year_overruled", "OpenAIChatGpt4", 1),
    get_ece("scotus", "year_overruled_few_shot", "OpenAIChatGpt4", 1),
    get_ece("scotus", "doctrinal_agreement", "OpenAIChatGpt4", 1),
    get_ece("scotus", "doctrinal_agreement_few_shot", "OpenAIChatGpt4", 1)),
  ChatGPT_SCOTUS = c(
    get_ece("scotus", "case_existence", "OpenAIChat", 1),
    get_ece("scotus", "case_existence_few_shot", "OpenAIChat", 1),
    get_ece("scotus", "court_id", "OpenAIChat", 1),
    get_ece("scotus", "court_id_few_shot", "OpenAIChat", 1),
    get_ece("scotus", "citation_retrieval", "OpenAIChat", 1),
    get_ece("scotus", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_ece("scotus", "majority_author", "OpenAIChat", 1),
    get_ece("scotus", "majority_author_few_shot", "OpenAIChat", 1),
    get_ece("scotus", "affirm_reverse", "OpenAIChat", 1),
    get_ece("scotus", "affirm_reverse_few_shot", "OpenAIChat", 1),
    get_ece("scotus", "year_overruled", "OpenAIChat", 1),
    get_ece("scotus", "year_overruled_few_shot", "OpenAIChat", 1),
    get_ece("scotus", "doctrinal_agreement", "OpenAIChat", 1),
    get_ece("scotus", "doctrinal_agreement_few_shot", "OpenAIChat", 1)),
  PaLM_SCOTUS = c(
    get_ece("scotus", "case_existence", "GooglePaLMCompletion", 1),
    get_ece("scotus", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_ece("scotus", "court_id", "GooglePaLMCompletion", 1),
    get_ece("scotus", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_ece("scotus", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_ece("scotus", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_ece("scotus", "majority_author", "GooglePaLMCompletion", 1),
    get_ece("scotus", "majority_author_few_shot", "GooglePaLMCompletion", 1),
    get_ece("scotus", "affirm_reverse", "GooglePaLMCompletion", 1),
    get_ece("scotus", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1),
    get_ece("scotus", "year_overruled", "GooglePaLMCompletion", 1),
    get_ece("scotus", "year_overruled_few_shot", "GooglePaLMCompletion", 1),
    get_ece("scotus", "doctrinal_agreement", "GooglePaLMCompletion", 1),
    get_ece("scotus", "doctrinal_agreement_few_shot", "GooglePaLMCompletion", 1)),
  Llama_SCOTUS = c(
    get_ece("scotus", "case_existence", "LlamaChat", 1),
    get_ece("scotus", "case_existence_few_shot", "LlamaChat", 1),
    get_ece("scotus", "court_id", "LlamaChat", 1),
    get_ece("scotus", "court_id_few_shot", "LlamaChat", 1),
    get_ece("scotus", "citation_retrieval", "LlamaChat", 1),
    get_ece("scotus", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_ece("scotus", "majority_author", "LlamaChat", 1),
    get_ece("scotus", "majority_author_few_shot", "LlamaChat", 1),
    get_ece("scotus", "affirm_reverse", "LlamaChat", 1),
    get_ece("scotus", "affirm_reverse_few_shot", "LlamaChat", 1),
    get_ece("scotus", "year_overruled", "LlamaChat", 1),
    get_ece("scotus", "year_overruled_few_shot", "LlamaChat", 1),
    get_ece("scotus", "doctrinal_agreement", "LlamaChat", 1),
    get_ece("scotus", "doctrinal_agreement_few_shot", "LlamaChat", 1)),
  ChatGPT4_COA = c(
    get_ece("coa", "case_existence", "OpenAIChatGpt4", 1),
    get_ece("coa", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_ece("coa", "court_id", "OpenAIChatGpt4", 1),
    get_ece("coa", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_ece("coa", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_ece("coa", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_ece("coa", "majority_author", "OpenAIChatGpt4", 1),
    get_ece("coa", "majority_author_few_shot", "OpenAIChatGpt4", 1),
    get_ece("coa", "affirm_reverse", "OpenAIChatGpt4", 1),
    get_ece("coa", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT_COA = c(
    get_ece("coa", "case_existence", "OpenAIChat", 1),
    get_ece("coa", "case_existence_few_shot", "OpenAIChat", 1),
    get_ece("coa", "court_id", "OpenAIChat", 1),
    get_ece("coa", "court_id_few_shot", "OpenAIChat", 1),
    get_ece("coa", "citation_retrieval", "OpenAIChat", 1),
    get_ece("coa", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_ece("coa", "majority_author", "OpenAIChat", 1),
    get_ece("coa", "majority_author_few_shot", "OpenAIChat", 1),
    get_ece("coa", "affirm_reverse", "OpenAIChat", 1),
    get_ece("coa", "affirm_reverse_few_shot", "OpenAIChat", 1),
    "-",
    "-",
    "-",
    "-"),
  PaLM_COA = c(
    get_ece("coa", "case_existence", "GooglePaLMCompletion", 1),
    get_ece("coa", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_ece("coa", "court_id", "GooglePaLMCompletion", 1),
    get_ece("coa", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_ece("coa", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_ece("coa", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_ece("coa", "majority_author", "GooglePaLMCompletion", 1),
    get_ece("coa", "majority_author_few_shot", "GooglePaLMCompletion", 1),
    get_ece("coa", "affirm_reverse", "GooglePaLMCompletion", 1),
    get_ece("coa", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1),
    "-",
    "-",
    "-",
    "-"),
  Llama_COA = c(
    get_ece("coa", "case_existence", "LlamaChat", 1),
    get_ece("coa", "case_existence_few_shot", "LlamaChat", 1),
    get_ece("coa", "court_id", "LlamaChat", 1),
    get_ece("coa", "court_id_few_shot", "LlamaChat", 1),
    get_ece("coa", "citation_retrieval", "LlamaChat", 1),
    get_ece("coa", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_ece("coa", "majority_author", "LlamaChat", 1),
    get_ece("coa", "majority_author_few_shot", "LlamaChat", 1),
    get_ece("coa", "affirm_reverse", "LlamaChat", 1),
    get_ece("coa", "affirm_reverse_few_shot", "LlamaChat", 1),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT4_USDC = c(
    get_ece("usdc", "case_existence", "OpenAIChatGpt4", 1),
    get_ece("usdc", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_ece("usdc", "court_id", "OpenAIChatGpt4", 1),
    get_ece("usdc", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_ece("usdc", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_ece("usdc", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_ece("usdc", "majority_author", "OpenAIChatGpt4", 1),
    get_ece("usdc", "majority_author_few_shot", "OpenAIChatGpt4", 1),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-"),
  ChatGPT_USDC = c(
    get_ece("usdc", "case_existence", "OpenAIChat", 1),
    get_ece("usdc", "case_existence_few_shot", "OpenAIChat", 1),
    get_ece("usdc", "court_id", "OpenAIChat", 1),
    get_ece("usdc", "court_id_few_shot", "OpenAIChat", 1),
    get_ece("usdc", "citation_retrieval", "OpenAIChat", 1),
    get_ece("usdc", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_ece("usdc", "majority_author", "OpenAIChat", 1),
    get_ece("usdc", "majority_author_few_shot", "OpenAIChat", 1),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-"),
  PaLM_USDC = c(
    get_ece("usdc", "case_existence", "GooglePaLMCompletion", 1),
    get_ece("usdc", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_ece("usdc", "court_id", "GooglePaLMCompletion", 1),
    get_ece("usdc", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_ece("usdc", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_ece("usdc", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_ece("usdc", "majority_author", "GooglePaLMCompletion", 1),
    get_ece("usdc", "majority_author_few_shot", "GooglePaLMCompletion", 1),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-"),
  Llama_USDC = c(
    get_ece("usdc", "case_existence", "LlamaChat", 1),
    get_ece("usdc", "case_existence_few_shot", "LlamaChat", 1),
    get_ece("usdc", "court_id", "LlamaChat", 1),
    get_ece("usdc", "court_id_few_shot", "LlamaChat", 1),
    get_ece("usdc", "citation_retrieval", "LlamaChat", 1),
    get_ece("usdc", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_ece("usdc", "majority_author", "LlamaChat", 1),
    get_ece("usdc", "majority_author_few_shot", "LlamaChat", 1),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-")
)
df5 <- df5 %>% mutate_all(linebreak, align = c("c"))
names(df5) <- df_column_names

# Create a LaTeX table
t5 <- df5 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Expected calibration error (ECE) across levels of the federal judiciary", escape = FALSE, linesep = "", table.env = "table*", label = "ECETable") %>%
  collapse_rows(columns = 1) %>%
  add_header_above(c(" " = 2, setNames(4, scotus), setNames(4, coa), setNames(4, usdc))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    footnote_order = c("alphabet", "general"),
    alphabet = c("1810-2022 (n=279)", "1796-2005 (n=5000)"),
    general = "\\\\\\\\ \\\\textit{Note:} Table reports expected calibration error between empirical hallucination rates and estimated conditional probabilities. Conditional probabilities are estimated by sampling 10 responses from the model at temperature 1 and assessing their agreement with the model\"s greedy response. Bootstrapped standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
t5 <- gsub("\\multirow{-2}{*}", "\\multirow{-2}{*}[1em]", t5, fixed = TRUE)
f5 <- str_glue("{TABLES_DIR}/ECETable.tex")
cat(paste(t5, collapse = "\n"), "\n", file = f5)
writeLines(str_glue("Saved table {f5}"))


#################################
# Table 6: ECE (temperature scaled)
#################################

# Create df
df6 <- data.frame(
  Task = tasks5,
  Prompt = prompts5,
  ChatGPT4_SCOTUS = c(
    get_ece("scotus", "case_existence", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "case_existence_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "court_id", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "court_id_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "majority_author", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "majority_author_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE)),
  ChatGPT_SCOTUS = c(
    get_ece("scotus", "case_existence", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "case_existence_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "court_id", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "court_id_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "majority_author", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "majority_author_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement", "OpenAIChat", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement_few_shot", "OpenAIChat", 1, rescale = TRUE)),
  PaLM_SCOTUS = c(
    get_ece("scotus", "case_existence", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "case_existence_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "court_id", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "court_id_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "majority_author", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "majority_author_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE)),
  Llama_SCOTUS = c(
    get_ece("scotus", "case_existence", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "case_existence_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "court_id", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "court_id_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "citation_retrieval_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "majority_author", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "majority_author_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "affirm_reverse_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "year_overruled_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement", "LlamaChat", 1, rescale = TRUE),
    get_ece("scotus", "doctrinal_agreement_few_shot", "LlamaChat", 1, rescale = TRUE)),
  ChatGPT4_COA = c(
    get_ece("coa", "case_existence", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "case_existence_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "court_id", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "court_id_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "majority_author", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "majority_author_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT_COA = c(
    get_ece("coa", "case_existence", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "case_existence_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "court_id", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "court_id_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "majority_author", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "majority_author_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse", "OpenAIChat", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse_few_shot", "OpenAIChat", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-"),
  PaLM_COA = c(
    get_ece("coa", "case_existence", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "case_existence_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "court_id", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "court_id_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "majority_author", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "majority_author_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-"),
  Llama_COA = c(
    get_ece("coa", "case_existence", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "case_existence_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "court_id", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "court_id_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "citation_retrieval_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "majority_author", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "majority_author_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse", "LlamaChat", 1, rescale = TRUE),
    get_ece("coa", "affirm_reverse_few_shot", "LlamaChat", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT4_USDC = c(
    get_ece("usdc", "case_existence", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "case_existence_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "court_id", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "court_id_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "majority_author", "OpenAIChatGpt4", 1, rescale = TRUE),
    get_ece("usdc", "majority_author_few_shot", "OpenAIChatGpt4", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-"),
  ChatGPT_USDC = c(
    get_ece("usdc", "case_existence", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "case_existence_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "court_id", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "court_id_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval_few_shot", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "majority_author", "OpenAIChat", 1, rescale = TRUE),
    get_ece("usdc", "majority_author_few_shot", "OpenAIChat", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-"),
  PaLM_USDC = c(
    get_ece("usdc", "case_existence", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "case_existence_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "court_id", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "court_id_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "majority_author", "GooglePaLMCompletion", 1, rescale = TRUE),
    get_ece("usdc", "majority_author_few_shot", "GooglePaLMCompletion", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-"),
  Llama_USDC = c(
    get_ece("usdc", "case_existence", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "case_existence_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "court_id", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "court_id_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "citation_retrieval_few_shot", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "majority_author", "LlamaChat", 1, rescale = TRUE),
    get_ece("usdc", "majority_author_few_shot", "LlamaChat", 1, rescale = TRUE),
    "-",
    "-",
    "-",
    "-",
    "-",
    "-")
)
df6 <- df6 %>% mutate_all(linebreak, align = c("c"))
names(df6) <- df_column_names

# Create a LaTeX table
t6 <- df6 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Temperature-scaled expected calibration error (ECE) across levels of the federal judiciary", escape = FALSE, linesep = "", table.env = "table*", label = "ECEScaledTable") %>%
  collapse_rows(columns = 1) %>%
  add_header_above(c(" " = 2, setNames(4, scotus), setNames(4, coa), setNames(4, usdc))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    footnote_order = c("alphabet", "general"),
    alphabet = c("1810-2022 (n=279)", "1796-2005 (n=5000)"),
    general = "\\\\\\\\ \\\\textit{Note:} Table reports temperature-scaled expected calibration error between empirical hallucination rates and estimated conditional probabilities. Conditional probabilities are estimated by sampling 10 responses from the model at temperature 1 and assessing their agreement with the model\"s greedy response. Bootstrapped standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
t6 <- gsub("\\multirow{-2}{*}", "\\multirow{-2}{*}[1em]", t6, fixed = TRUE)
f6 <- str_glue("{TABLES_DIR}/ECEScaledTable.tex")
cat(paste(t6, collapse = "\n"), "\n", file = f6)
writeLines(str_glue("Saved table {f6}"))


#################################
# Table 7: Abstention rates (all tasks)
#################################

# Create df
tasks7 <- c(tasks, tasks2, paste0("Doctrinal agreement", footnote_marker_alphabet(2, format = "latex", double_escape = FALSE)), paste0("Doctrinal agreement", footnote_marker_alphabet(2, format = "latex", double_escape = FALSE)))
prompts7 <- c(prompts, prompts2, "Zero-shot", "Few-shot")
df7 <- data.frame(
  Task = tasks7,
  Prompt = prompts7,
  ChatGPT4_SCOTUS = c(
    get_no_answer_rate("scotus", "case_existence", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "court_id", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "majority_author", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "majority_author_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "affirm_reverse", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "quotation", "OpenAIChatGpt4", -99),
    get_no_answer_rate("scotus", "quotation_few_shot", "OpenAIChatGpt4", -99),
    get_no_answer_rate("scotus", "cited_precedent", "OpenAIChatGpt4", -99),
    get_no_answer_rate("scotus", "cited_precedent_few_shot", "OpenAIChatGpt4", -99),
    get_no_answer_rate("scotus", "year_overruled", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "year_overruled_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement", "OpenAIChatGpt4", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement_few_shot", "OpenAIChatGpt4", 1)),
  ChatGPT_SCOTUS = c(
    get_no_answer_rate("scotus", "case_existence", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "case_existence_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "court_id", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "court_id_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "citation_retrieval", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "majority_author", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "majority_author_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "affirm_reverse", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "affirm_reverse_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "quotation", "OpenAIChat", -99),
    get_no_answer_rate("scotus", "quotation_few_shot", "OpenAIChat", -99),
    get_no_answer_rate("scotus", "cited_precedent", "OpenAIChat", -99),
    get_no_answer_rate("scotus", "cited_precedent_few_shot", "OpenAIChat", -99),
    get_no_answer_rate("scotus", "year_overruled", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "year_overruled_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement", "OpenAIChat", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement_few_shot", "OpenAIChat", 1)),
  PaLM_SCOTUS = c(
    get_no_answer_rate("scotus", "case_existence", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "court_id", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "majority_author", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "majority_author_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "affirm_reverse", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "quotation", "GooglePaLMCompletion", -99),
    get_no_answer_rate("scotus", "quotation_few_shot", "GooglePaLMCompletion", -99),
    get_no_answer_rate("scotus", "cited_precedent", "GooglePaLMCompletion", -99),
    get_no_answer_rate("scotus", "cited_precedent_few_shot", "GooglePaLMCompletion", -99),
    get_no_answer_rate("scotus", "year_overruled", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "year_overruled_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement", "GooglePaLMCompletion", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement_few_shot", "GooglePaLMCompletion", 1)),
  LLama_SCOTUS = c(
    get_no_answer_rate("scotus", "case_existence", "LlamaChat", 1),
    get_no_answer_rate("scotus", "case_existence_few_shot", "LlamaChat", 1),
    get_no_answer_rate("scotus", "court_id", "LlamaChat", 1),
    get_no_answer_rate("scotus", "court_id_few_shot", "LlamaChat", 1),
    get_no_answer_rate("scotus", "citation_retrieval", "LlamaChat", 1),
    get_no_answer_rate("scotus", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_no_answer_rate("scotus", "majority_author", "LlamaChat", 1),
    get_no_answer_rate("scotus", "majority_author_few_shot", "LlamaChat", 1),
    get_no_answer_rate("scotus", "affirm_reverse", "LlamaChat", 1),
    get_no_answer_rate("scotus", "affirm_reverse_few_shot", "LlamaChat", 1),
    get_no_answer_rate("scotus", "quotation", "LlamaChat", -99),
    get_no_answer_rate("scotus", "quotation_few_shot", "LlamaChat", -99),
    get_no_answer_rate("scotus", "cited_precedent", "LlamaChat", -99),
    get_no_answer_rate("scotus", "cited_precedent_few_shot", "LlamaChat", -99),
    get_no_answer_rate("scotus", "year_overruled", "LlamaChat", 1),
    get_no_answer_rate("scotus", "year_overruled_few_shot", "LlamaChat", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement", "LlamaChat", 1),
    get_no_answer_rate("scotus", "doctrinal_agreement_few_shot", "LlamaChat", 1)),
  ChatGPT4_COA = c(
    get_no_answer_rate("coa", "case_existence", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "court_id", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "majority_author", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "majority_author_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "affirm_reverse", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "affirm_reverse_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("coa", "quotation", "OpenAIChatGpt4", -99),
    get_no_answer_rate("coa", "quotation_few_shot", "OpenAIChatGpt4", -99),
    get_no_answer_rate("coa", "cited_precedent", "OpenAIChatGpt4", -99),
    get_no_answer_rate("coa", "cited_precedent_few_shot", "OpenAIChatGpt4", -99),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT_COA = c(
    get_no_answer_rate("coa", "case_existence", "OpenAIChat", 1),
    get_no_answer_rate("coa", "case_existence_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("coa", "court_id", "OpenAIChat", 1),
    get_no_answer_rate("coa", "court_id_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("coa", "citation_retrieval", "OpenAIChat", 1),
    get_no_answer_rate("coa", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("coa", "majority_author", "OpenAIChat", 1),
    get_no_answer_rate("coa", "majority_author_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("coa", "affirm_reverse", "OpenAIChat", 1),
    get_no_answer_rate("coa", "affirm_reverse_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("coa", "quotation", "OpenAIChat", -99),
    get_no_answer_rate("coa", "quotation_few_shot", "OpenAIChat", -99),
    get_no_answer_rate("coa", "cited_precedent", "OpenAIChat", -99),
    get_no_answer_rate("coa", "cited_precedent_few_shot", "OpenAIChat", -99),
    "-",
    "-",
    "-",
    "-"),
  PaLM_COA = c(
    get_no_answer_rate("coa", "case_existence", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "court_id", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "majority_author", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "majority_author_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "affirm_reverse", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "affirm_reverse_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("coa", "quotation", "GooglePaLMCompletion", -99),
    get_no_answer_rate("coa", "quotation_few_shot", "GooglePaLMCompletion", -99),
    get_no_answer_rate("coa", "cited_precedent", "GooglePaLMCompletion", -99),
    get_no_answer_rate("coa", "cited_precedent_few_shot", "GooglePaLMCompletion", -99),
    "-",
    "-",
    "-",
    "-"),
  LLama_COA = c(
    get_no_answer_rate("coa", "case_existence", "LlamaChat", 1),
    get_no_answer_rate("coa", "case_existence_few_shot", "LlamaChat", 1),
    get_no_answer_rate("coa", "court_id", "LlamaChat", 1),
    get_no_answer_rate("coa", "court_id_few_shot", "LlamaChat", 1),
    get_no_answer_rate("coa", "citation_retrieval", "LlamaChat", 1),
    get_no_answer_rate("coa", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_no_answer_rate("coa", "majority_author", "LlamaChat", 1),
    get_no_answer_rate("coa", "majority_author_few_shot", "LlamaChat", 1),
    get_no_answer_rate("coa", "affirm_reverse", "LlamaChat", 1),
    get_no_answer_rate("coa", "affirm_reverse_few_shot", "LlamaChat", 1),
    get_no_answer_rate("coa", "quotation", "LlamaChat", -99),
    get_no_answer_rate("coa", "quotation_few_shot", "LlamaChat", -99),
    get_no_answer_rate("coa", "cited_precedent", "LlamaChat", -99),
    get_no_answer_rate("coa", "cited_precedent_few_shot", "LlamaChat", -99),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT4_USDC = c(
    get_no_answer_rate("usdc", "case_existence", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "case_existence_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "court_id", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "court_id_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "citation_retrieval", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "citation_retrieval_few_shot", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "majority_author", "OpenAIChatGpt4", 1),
    get_no_answer_rate("usdc", "majority_author_few_shot", "OpenAIChatGpt4", 1),
    "-",
    "-",
    get_no_answer_rate("usdc", "quotation", "OpenAIChatGpt4", -99),
    get_no_answer_rate("usdc", "quotation_few_shot", "OpenAIChatGpt4", -99),
    get_no_answer_rate("usdc", "cited_precedent", "OpenAIChatGpt4", -99),
    get_no_answer_rate("usdc", "cited_precedent_few_shot", "OpenAIChatGpt4", -99),
    "-",
    "-",
    "-",
    "-"),
  ChatGPT_USDC = c(
    get_no_answer_rate("usdc", "case_existence", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "case_existence_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "court_id", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "court_id_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "citation_retrieval", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "citation_retrieval_few_shot", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "majority_author", "OpenAIChat", 1),
    get_no_answer_rate("usdc", "majority_author_few_shot", "OpenAIChat", 1),
    "-",
    "-",
    get_no_answer_rate("usdc", "quotation", "OpenAIChat", -99),
    get_no_answer_rate("usdc", "quotation_few_shot", "OpenAIChat", -99),
    get_no_answer_rate("usdc", "cited_precedent", "OpenAIChat", -99),
    get_no_answer_rate("usdc", "cited_precedent_few_shot", "OpenAIChat", -99),
    "-",
    "-",
    "-",
    "-"),
  PaLM_USDC = c(
    get_no_answer_rate("usdc", "case_existence", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "case_existence_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "court_id", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "court_id_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "citation_retrieval", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "citation_retrieval_few_shot", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "majority_author", "GooglePaLMCompletion", 1),
    get_no_answer_rate("usdc", "majority_author_few_shot", "GooglePaLMCompletion", 1),
    "-",
    "-",
    get_no_answer_rate("usdc", "quotation", "GooglePaLMCompletion", -99),
    get_no_answer_rate("usdc", "quotation_few_shot", "GooglePaLMCompletion", -99),
    get_no_answer_rate("usdc", "cited_precedent", "GooglePaLMCompletion", -99),
    get_no_answer_rate("usdc", "cited_precedent_few_shot", "GooglePaLMCompletion", -99),
    "-",
    "-",
    "-",
    "-"),
  LLama_USDC = c(
    get_no_answer_rate("usdc", "case_existence", "LlamaChat", 1),
    get_no_answer_rate("usdc", "case_existence_few_shot", "LlamaChat", 1),
    get_no_answer_rate("usdc", "court_id", "LlamaChat", 1),
    get_no_answer_rate("usdc", "court_id_few_shot", "LlamaChat", 1),
    get_no_answer_rate("usdc", "citation_retrieval", "LlamaChat", 1),
    get_no_answer_rate("usdc", "citation_retrieval_few_shot", "LlamaChat", 1),
    get_no_answer_rate("usdc", "majority_author", "LlamaChat", 1),
    get_no_answer_rate("usdc", "majority_author_few_shot", "LlamaChat", 1),
    "-",
    "-",
    get_no_answer_rate("usdc", "quotation", "LlamaChat", -99),
    get_no_answer_rate("usdc", "quotation_few_shot", "LlamaChat", -99),
    get_no_answer_rate("usdc", "cited_precedent", "LlamaChat", -99),
    get_no_answer_rate("usdc", "cited_precedent_few_shot", "LlamaChat", -99),
    "-",
    "-",
    "-",
    "-")
)
df7 <- df7 %>% mutate_all(linebreak, align = c("c"))
names(df7) <- df_column_names

# Create a LaTeX table
t7 <- df7 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Absention rates across levels of the federal judiciary (resource-aware tasks)", escape = FALSE, linesep = "", table.env = "table*", label = "NonResponseTable") %>%
  collapse_rows(columns = 1) %>%
  add_header_above(c(" " = 2, setNames(4, scotus), setNames(4, coa), setNames(4, usdc))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    footnote_order = c("alphabet", "general"),
    alphabet = c("1810-2022 (n=279)", "1796-2005 (n=5000)"),
    general = "\\\\\\\\ \\\\textit{Note:} Table reports model abstention rates. Standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
t7 <- gsub("\\multirow{-2}{*}", "\\multirow{-2}{*}[1em]", t7, fixed = TRUE)
f7 <- str_glue("{TABLES_DIR}/NonResponseTable.tex")
cat(paste(t7, collapse = "\n"), "\n", file = f7)
writeLines(str_glue("Saved table {f7}"))


#################################
# Table 8: Hallucination rates (fake existence task)
#################################

# Create df
tasks8 <- c("False existence")
prompts8 <- c("Zero-shot")
df8 <- data.frame(
  Task = tasks8,
  Prompt = prompts8,
  ChatGPT4_SCOTUS = c(
    get_hr("scotus", "fake_case_existence", "OpenAIChatGpt4", 1)
  ),
  ChatGPT_SCOTUS = c(
    get_hr("scotus", "fake_case_existence", "OpenAIChat", 1)
  ),
  PaLM_SCOTUS = c(
    get_hr("scotus", "fake_case_existence", "GooglePaLMCompletion", 1)
  ),
  Llama_SCOTUS = c(
    get_hr("scotus", "fake_case_existence", "LlamaChat", 1)
  ),
  ChatGPT4_COA = c(
    get_hr("coa", "fake_case_existence", "OpenAIChatGpt4", 1)
  ),
  ChatGPT_COA = c(
    get_hr("coa", "fake_case_existence", "OpenAIChat", 1)
  ),
  PaLM_COA = c(
    get_hr("coa", "fake_case_existence", "GooglePaLMCompletion", 1)
  ),
  Llama_COA = c(
    get_hr("coa", "fake_case_existence", "LlamaChat", 1)
  ),
  ChatGPT4_USDC = c(
    get_hr("usdc", "fake_case_existence", "OpenAIChatGpt4", 1)
  ),
  ChatGPT_USDC = c(
    get_hr("usdc", "fake_case_existence", "OpenAIChat", 1)
  ),
  PaLM_USDC = c(
    get_hr("usdc", "fake_case_existence", "GooglePaLMCompletion", 1)
  ),
  Llama_USDC = c(
    get_hr("usdc", "fake_case_existence", "LlamaChat", 1)
  )
)
df8 <- df8 %>% mutate_all(linebreak, align = c("c"))
names(df8) <- df_column_names

# Create a LaTeX table
t8 <- df8 %>%
  kable(format = "latex", booktabs = TRUE, caption = "Hallucination rates across levels of the federal judiciary (fake existence task)", escape = FALSE, linesep = "", table.env = "table*", label = "FakeExistenceTable") %>%
  add_header_above(c(" " = 2, setNames(4, "SCOTUS\n(1794-2015; n=1000)"), setNames(4, "USCOA\n(1895-2019; n=1000)"), setNames(4, "USDC\n(1932-2019; n=5000)"))) %>%
  kable_styling(latex_options = c("scale_down"), full_width = F) %>%
  footnote(
    general = "\\\\textit{Note:} Table reports estimated hallucination rates. Standard errors are shown in parentheses.",
    general_title = "",
    threeparttable = TRUE,
    footnote_as_chunk = TRUE,
    escape = FALSE
  )
f8 <- str_glue("{TABLES_DIR}/FakeExistenceTable.tex")
cat(paste(t8, collapse = "\n"), "\n", file = f8)
writeLines(str_glue("Saved table {f8}"))
