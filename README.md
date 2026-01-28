# How Does Unfaithful Reasoning Emerge from Autoregressive Training?  
### Anonymous code release for ICML 2026 submission

This repository contains the implementation of the experiments described in the paper:

> "How Does Unfaithful Reasoning Emerge from Autoregressive Training?  
> A Study of Synthetic Experiments"  
> Anonymous submission to ICML 2026.

This work investigates how unfaithful chain-of-thought behaviors—such as mixed, skip-step reasoning and self-verification—emerge in autoregressively trained Transformers by using a controlled noisy modular-arithmetic task with dedicated faithfulness metrics to reveal phase transitions between reasoning modes.


### Usage
```bash
conda create -n ood python=3.10
conda activate ood
pip install -r requirements.txt
pip install -e .
```

To run Experiments on the minimal data format used in the paper (for example), you can execute the following command(s):
```
bash arithmetic_experiments/minimal_task_experiment/main_wop.sh
```


### Structure of the important files
```text
CoT-faithfulness-synthetic-arithmetic/
├── __init__.py
├── config.yaml
├── requirements.txt
├── arithmetic_experiments/
│   ├── __init__.py
│   ├── minimal_task_experiment/
│   │   ├── main_wop.sh
│   │   ├── main_wop.py
│   │   └── ... (other files omitted)
│   ├── parentheses_task_experiment/
│   │   └── ... (similar structure omitted)
│   └── extended_task_experiment/
│       └── ... (similar structure omitted)
├── tasks/
│   ├── minimal_task/
│   │   ├── Generator_wop.py
│   │   ├── modular_data_generation_wop.py
│   │   └── ... (other files omitted)
│   ├── parentheses_task/
│   │   └── ... (similar structure omitted)
│   └── extended_task/
│       └── ... (similar structure omitted)
└── model/
    └── model.py
```


## License

To preserve anonymity for double-blind review, the license information is temporarily omitted.
A license will be added after the review process.