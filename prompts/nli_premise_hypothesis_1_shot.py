PROMPT = """Analyze pairs of statements about historical U.S. Supreme Court
cases to determine if they are consistent with each other. Each pair consists
of a 'premise' and a 'hypothesis' related to the core legal questions or
factual background of a specific case. Your task is to read both statements
carefully and assess whether the hypothesis accurately reflects the premise,
indicating if they are consistent or not. Provide a brief explanation before
reaching a conclusion. Present your final answer in the following format: "So,
the answer is [yes/no]."

###

Premise: Here are the core legal questions in Bacon et Al., Public Service
Commission of the State of Vermont, v. Rutland Railroad Company, 232 U.S. 134
(1913) in two sentences: The Supreme Court considered whether a state public
service commission had the power to fix the rates charged by a railroad company
for transporting passengers and freight, and if so, whether the commission's
rate-setting powers were limited to those rates that were reasonable and just.

Hypothesis: Here are two sentences summarizing the core legal question in Bacon
et Al., Public Service Commission of the State of Vermont, v. Rutland Railroad
Company, 232 U.S. 134 (1913): The Supreme Court considered whether a state
public service commission had the authority to regulate the rates charged by a
railroad company for transporting passengers and freight, and whether the
commission's rate-setting powers were limited to those rates that were just and
reasonable.

Q: Is the premise consistent with the hypothesis?

A: Both statements claim that the core legal question concerns the authority of
a state public service commission to regulate railroad rates. So, the answer is
yes.


###

Premise: {premise}

Hypothesis: {hypothesis}

Q: Is the premise consistent with the hypothesis?
"""
