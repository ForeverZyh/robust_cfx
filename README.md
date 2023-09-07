# Robust CFX

## Prerequisites
```
conda create -n robust_cfx python=3.8
conda activate robust_cfx
pip install -r requirements.txt
```

## Training models
Run `train.py`. Specify the path to save the model parameters, and optionally specify IBP (default) or standard training with the model keyword, e.g., 

`python train.py standard_model --model Standard`

## Evaluating models
Run `eval.py`. Specify the path to the model parameters, e.g., `python eval.py models/standard_model.pt`.

`eval.py` currently outputs the fraction of test samples that are certifiably robust (I think) AND whose counterexamples are certifiably robust.

## Counterfactual generation
Currently [Wachter et al.'s](https://arxiv.org/abs/1711.00399) method is the only CFX generation technique supported (we use the [Alibi library implementation](https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html)).
