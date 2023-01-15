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
|   |   flan.py                  # Currently used FLAN code for hypothesis generation
|   |   dataset.py               # HF dataset loader and preprocessing
|   |   rule_filters.py          # NELLIE: Various rule filter classes
|   |   problog_fns.py           # NELLIE: Code for problog functionality
|   |   sbert.py                 # SBERT encoder fine-tuning code/script
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
    |   eval_retrieval.py        # Script to evaluate dialogue_retrieval.py
```

# Installation

# Misc
