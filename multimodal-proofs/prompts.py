entailment_definition = 'Textual entailment is defined as a directional relation between two text fragments, called text (t, the entailing text), and hypothesis (h, the entailed text), so that a human being, with common understanding of language and common background knowledge, can infer that h is most likely true on the basis of the content of t.'

declaritivize_prompt = """
Combine the following question-answer pair into a single declarative statement. 
QUESTION: "{q}"
ANSWER: "{a}"
STATEMENT: 
"""

inference_preamble = entailment_definition + \
"""
\nWrite a set of five hypotheses that relate to the FACT and are specifically entailed by the specified dialogue line in JSON format, i.e. {"1": "<answer here>", "2": "<answer here>", ...} and nothing else.
"""

inference_prompt=""" 
\nFACT: "{h}" 
DIALOGUE: 
```
{d}
```
LINE ({l}) ENTAILMENTS:
"""

branch_preamble = entailment_definition + \
"""
\nWrite two facts that are entailed by the dialogue that, together, make the hypothesis true. Write your answer in JSON format, i.e. {"1": "<fact 1>", "2": "<fact 2>"} and nothing else.
"""

branch_prompt = """
\nHYPOTHESIS: "{h}"
DIALOGUE:
```
{d}
```
FACTS:
"""

toq_prompt = """
Convert the following statement into a "yes" or "no" question, and then rewrite the question with the names replaced with "person" or "people".
STATEMENT: "{h}"
QUESTION: 
"""
