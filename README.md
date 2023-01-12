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
└───flan
|   |   extract_chains.py        # Primary FLAN script
|   |   timers.py                # Timing helper functions for FLAN script
|   └───utils
|       |    __init__.py         # Additional helper functions for FLAN script
|
└───src
|   |   dataset.py               # Dataset structuring classes
|   |   engine.py                # Main NELLIE class for tree generation
|   |   fact_retrieval.py        # NELLIE FAISS search index
|   |   hf_dataset.py            # HF dataset loader and preprocessing
|   |   lm_engine.py             # Language model for the problog model
|   |   metrics.py               # Evaluation function
|   |   rule_filters.py          # Various rule filter classes
|   |   problog_fns.py           # Code for problog functionality
|   |   sbert.py                 # SBERT encoder fine-tuning code
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
|       |    hf_utils.py         # Seq2seq model and data classes and functions
|       |    random_mask.py      # Masking helper functions
|       |    read_data.py        # Text parsing helper functions
|       |    utils.py            # General helper function file
|
└───tools
    |   gen_hypotheses.py        # QA pair to hypothesis conversion script
    |   run_engine.py            # Inference script
    |   run_flan.py              # Alternative to flan/ and gen_hypotheses scripts
```

# Installation

# Misc
