entailment_definition = 'Textual entailment is defined as a directional relation between two text fragments, called text (t, the entailing text), and hypothesis (h, the entailed text), so that a human being, with common understanding of language and common background knowledge, can infer that h is most likely true on the basis of the content of t.'

declarativize_prompt = """
Combine the following question-answer pair into a single declarative statement. 
QUESTION: "{q}"
ANSWER: "{a}"
STATEMENT: 
"""
'''
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


inference_preamble_a = """
You are a fact-checking expert that determines whether a hypothesis about a TV show is true or false.

To see if the hypothesis is true, you write five inferences about the show that best support the hypothesis. 

Write your facts in JSON format, i.e. {\"1\": \"<answer here>\", \"2\": \"<answer here>\", ...} and nothing else.
"""

inference_prompt_a = """
\nHYPOTHESIS: \"{h}\" 
SCENE: 
```
{d}
```
RESPONSE:
"""
'''

inference_preamble = """
You are a fact-checking expert that determines whether a hypothesis about a TV show is true or false.

To see if the hypothesis is true, you write five inferences about the show that best support the hypothesis.

Write your facts in JSON format, i.e. {\"1\": \"<answer here>\", \"2\": \"<answer here>\", ...} and nothing else.
"""

inference_prompt = """
\nHYPOTHESIS: \"{h}\" 
SCENE: 
```
{d}
```
RESPONSE:
"""

'''
branch_a_preamble = entailment_definition + \
"""
\nWrite two facts that are entailed by the dialogue that, together, make the hypothesis true. Write your answer in JSON format, i.e. {"1": "<fact 1>", "2": "<fact 2>"} and nothing else.
"""

branch_a_prompt = """
\nHYPOTHESIS: "{h}"
DIALOGUE:
```
{d}
```
FACTS:
"""
'''


branch_a_preamble = """
Break down the following sentence into two, simpler sentences.

Your output should be in JSON format, i.e. {\"statement1\": \"<answer here>\", \"statement2\": \"<answer here>\"}, and nothing else.
"""

branch_a_prompt = """
\n\nSENTENCE: 
"{h}\"

RESPONSE: 
"""

branch_b_preamble = """
You are a fact-checking expert that determines whether a statement about a TV show is true or false.

To see if the hypothesis is true, you write two hypotheses about the show that, together, would make the statement true. Write your answer in JSON format, i.e. {"1": "<fact 1>", "2": "<fact 2>"} and nothing else.
"""

branch_b_prompt = """
\n\nSTATEMENT: \"{h}\" 

SCENE:
```
{d}
```

HYPOTHESES:
"""

verify_i_preamble = entailment_definition + \
"""
\nWrite two facts that are entailed by the dialogue that, together, make the hypothesis true. Write your answer in JSON format, i.e. {"1": "<fact 1>", "2": "<fact 2>"} and nothing else.
"""

verify_i_prompt = """
\nHYPOTHESIS: "{h}"
DIALOGUE:
```
{d}
```
FACTS:
"""

verify_b_preamble = entailment_definition + \
"""
\nWrite two facts that are entailed by the dialogue that, together, make the hypothesis true. Write your answer in JSON format, i.e. {"1": "<fact 1>", "2": "<fact 2>"} and nothing else.
"""

verify_b_prompt = """
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

verify_inf_preamble = """
You are an expert reasoning system. Given a transcript and a statement, you write \"YES\" if the statement is directly entailed by the transcript and \"NO\" if it is not.
"""

verify_inf_prompt = """
\n\nTRANSCRIPT: \n```\n{d}\n```\n\nHYPOTHESIS: \"{h}\" \n\nRESPONSE:
"""