dataset=$1

if [ -z "$dataset" ]; then
    echo "Please provide a dataset"
    exit 1
fi

modeldir="trained_models"
logdir="logs"
cfxdir="saved_cfxs"
nummodels=10
modelcnt=$(($nummodels - 1))

# step 1: generate CFX and compute delta-robustness
for i in $(seq 0 $modelcnt)
do
    echo $i
    modelname=$dataset"IBP"$i 
    if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
        python eval.py $modelname $dataset --save_dir $modeldir \
           --cfx_save_dir $cfxdir --log_save_dir $logdir --skip_milp --finetune
    else 
        python eval.py $modelname $dataset --save_dir $modeldir \
           --cfx_save_dir $cfxdir --log_save_dir $logdir --skip_milp
    fi
done

# step 2: convert logs to processed CSV
python scripts/convert_logs.py --log_dir $logdir --skip_milp --target_datasets $dataset --filename $dataset"_robustness.csv"

# step 3: compute cross-model-validity
logdir="validity_logs"
if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
    python scripts/cross_model_validity.py $dataset"IBP" $dataset "ours" --save_dir $modeldir \
          --cfx_save_dir $cfxdir --log_save_dir $logdir --model_cnt $nummodels --finetune
else 
    python scripts/cross_model_validity.py $dataset"IBP" $dataset "ours" --save_dir $modeldir \
        --cfx_dir $cfxdir --log_save_dir $logdir --model_cnt $nummodels
fi

# step 4: generate final data
python scripts/convert_validity_logs.py --log_dir $logdir --target_datasets $dataset --filename $dataset"_validity.csv" 