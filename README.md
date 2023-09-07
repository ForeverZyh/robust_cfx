# Robust CFX

## Prerequisites
```
conda create -n robust_cfx python=3.8
conda activate robust_cfx
pip install -r requirements.txt
```

## Training models
Run `train.py`. Specify the path to save the model parameters, and optionally specify IBP (default) or standard training with the model keyword, e.g., 

To train with `wachter` 

```commandline
python train.py standard_model --model Standard --cfx wachter --epoch 20
python train.py IBP_model --model IBP --cfx wachter --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-1
python train.py IBP_inc_model --model IBP --cfx wachter --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-1 --inc_regenerate
```

To train with `proto`, specify the `--onehot` flag.

```commandline
python train.py standard_model_proto --model Standard --cfx proto --onehot --epoch 20
python train.py IBP_model_proto --model IBP --cfx proto --onehot --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-1
python train.py IBP_inc_model_proto --model IBP --cfx proto --onehot --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-1 --inc_regenerate
```

## Evaluating models
Run `eval.py`. Specify the path to the model parameters, e.g., `python eval.py models/standard_model.pt`.

`eval.py` currently outputs the fraction of test samples that are certifiably robust (I think) AND whose counterexamples are certifiably robust.

```commandline
python eval.py models/standard_model.pt --cfx wachter --epsilon 1e-3 --bias_epsilon 1e-2
python eval.py models/IBP_model.pt --cfx wachter --epsilon 1e-3 --bias_epsilon 1e-2
python eval.py models/IBP_inc_model.pt --cfx wachter --epsilon 1e-3 --bias_epsilon 1e-2
python eval.py models/standard_model_proto.pt --cfx proto --onehot --epsilon 1e-2 --bias_epsilon 1e-1
python eval.py models/IBP_model_proto.pt --cfx proto --onehot --epsilon 1e-2 --bias_epsilon 1e-1
python eval.py models/IBP_inc_model_proto.pt --cfx proto --onehot --epsilon 1e-2 --bias_epsilon 1e-1
```

## Counterfactual generation
Currently [Wachter et al.'s](https://arxiv.org/abs/1711.00399) method is the only CFX generation technique supported (we use the [Alibi library implementation](https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html)).
