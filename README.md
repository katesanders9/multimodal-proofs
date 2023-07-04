# Overview
Code for multimodal neuro-symbolic proof generation for TV shows

# Code directory

```
multimodal-proofs
|
|   README.md                    # Repository documentation
|
└───cfg
|   |   config.yaml              # Config file for tools/ scripts
|
└───src
|   |   dialogue_retrieval.py    # Dialogue retrieval engine
|   |   filters.py               # Filters for entailment
|   |   flan.py                  # Currently used FLAN code for hypothesis generation
|   |   dataset.py               # HF dataset loader and preprocessing
|   |   rule_generators.py       # NELLIE: Helps define network for problog model
|   |   problog_fns.py           # NELLIE: Code for problog functionality
|   |   sbert.py                 # SBERT encoder fine-tuning code/script
|   |   search.py                # Generate dialogue tree
|   |   vision.py                # Vision module
|   |
|   └───problog
|   |   |    problog_rules.pl    # Main recursive problog proof rule
|   |   |    proof_methods.pl    # Additional problog proof rules
|   |   |    retrieval.pl        # Problog retrieval rules
|   |   |    timer.pl            # Problog timing rules
|   |   |    utils.pl            # Problog helper rules
|   |
|   └───utils
|       |    __init__.py
|       |    utils.py            # General helper function file
|
└───tools
    |   eval_entailment.py       # Script to evaluate filters.py
    |   eval_retrieval.py        # Script to evaluate dialogue_retrieval.py
    |   eval_search.py           # Script to evaluate dialogue tree generation
```

# Installation

# Misc
