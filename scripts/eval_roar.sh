dataset=$1

if [ -z "$dataset" ]; then
    echo "Please provide a dataset"
    exit 1
fi

modeldir="trained_models"
logdir="validity_logs"
cfxdir="saved_cfxs"
numtorun=500

# step 1: generate CFX and compute delta-robustness
modelname=$dataset"Standard"
if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
    python scripts/inn_exp.py $modelname $dataset --save_dir $modeldir \
        --cfx_dir $cfxdir --log_dir $logdir --num_to_run $numtorun --finetune --target_idxs 0 1
else 
    python scripts/inn_exp.py $modelname $dataset --save_dir $modeldir \
        --cfx_dir $cfxdir --log_dir $logdir --num_to_run $numtorun
fi

# step 2: generate final validity data
python scripts/convert_validity_logs.py --log_dir $logdir --target_datasets $dataset --filename $dataset"_validity.csv" 

# step 3: delta-robustness 
nummodels=10
modelcnt=$(($nummodels - 1))
logdir="logs"
for i in $(seq 0 $modelcnt)
do
    modelname=$dataset"Standard"$i
    if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
        python eval.py $modelname $dataset --save_dir $modeldir --num_to_run $numtorun \
            --cfx_save_dir $cfxdir --log_save_dir $logdir --skip_milp --cfx_technique "roar" --finetune
    else 
        python eval.py $modelname $dataset --save_dir $modeldir --num_to_run $numtorun \
            --cfx_save_dir $cfxdir --log_save_dir $logdir --cfx_technique "roar" --skip_milp
    fi
done

python scripts/convert_logs.py --log_dir $logdir --skip_milp --target_datasets $dataset --filename $dataset"_robustness.csv"
