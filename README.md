# Robust Counterfactual Explanations (CEs)

## Prerequisites
```
conda create -n robust_cfx python=3.8
conda activate robust_cfx
pip install -r requirements.txt
```

## Reproducing our experiments
**Step 0** Set hyperparameters: our default hyperparameters are stored in JSON files in the `assets` directory. If desired, change any hyperparameters there first. Certain parameters will be automatically adjusted depending on parameters passed to `train.py`, e.g., learning rate is lowered for finetuning and `lambda` parameters are adjusted for Standard (non-CounterNet) training.

**Step 1** Train models using `scripts/train.sh`
- Provide the dataset (heloc, taiwan, student, who, ctg) as input, e.g., `./scripts/train.sh student`
- *Tip:* if you want to test our method but are short on time, change the number of epochs in line 11 and the number of models in line 13
- *Tip:* This script calculates delta-robustness, random initialization, and finetuning robustness (for CTG and WHO). If you want to try LOO, you can uncomment the bottom section of the script. You'll need to update `scripts/eval.sh` to perform the evaluation, though.

**Step 2** Generate CEs and evaluate robustness using `scripts/eval.sh`
- If you adjusted the number of models when training models, edit `nummodels` on line 11.

**Step 3** If desired, reproduce our baselines.
- ROAR
  - First, run `scripts/train_standard.sh` to generate models.
  - Then, run `scripts/eval_roar.sh` to generate CEs
- SNS
  - If not done yet, run `scripts/train_standard.sh` to generate models.
  - Then, run `scripts/eval_sns.sh` to generate CEs
- CounterNet
  - Run `scripts/train_cn.sh` to generate models.
  - Then, run `scripts/eval_cn.sh` to generate CEs

## Training models
Run `train.py`. Specify the name of the model and the dataset. Optionally, customize other parameters (see `train.py` for details). 

E.g., execute `python train.py StudentIBP0 student` to run VeriTraCER, `python train.py StudentCN0 student --model CN` to run CounterNet without robustness modifications, or `python train.py StudentStandard0 student --model Standard` to train a standard model (i.e., a traditional model with no (meaningful) CFX generator). 

## Evaluating models
The core of the delta-robustness evalutation (and CFX generation) happens in `eval.py`. 
Cross-model-validity is analyzed in `scripts/cross_model_validity.py`. Then, the scripts `scripts/convert_logs` and `scripts/convert_validity_logs.py` postprocess the generated logs into a single CSV file for delta-robustness and cross-model-validity, respectively.
