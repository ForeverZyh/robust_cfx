dataset=$1

if [ -z "$dataset" ]; then
    echo "Please provide a dataset"
    exit 1
fi

modeldir="trained_models"
logdir="validity_logs"
cfxdir="saved_cfxs"
numtorun=500

nummodels=10
modelcnt=$(($nummodels - 1))

# Step 0: Convert pytorch models to tensorflow
for i in $(seq 0 $modelcnt)
do
    if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
        python sns/convert_pytorch_tf.py $dataset"Standard"$i $dataset --finetune
    else 
        python sns/convert_pytorch_tf.py $dataset"Standard"$i $dataset 
    fi    
done  

# step 1: generate CFX
modelname=$dataset"Standard"
for i in $(seq 0 $modelcnt)
do
    python sns/run_sns.py $modelname$i $dataset --technique l1 --cfx_save_dir $cfxdir --num_to_run $numtorun
done

# step 2: cross-model validity
if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
    python sns/sns_cross_model.py $modelname $dataset "l1" --model_cnt $nummodels \
            --cfx_dir $cfx_dir --num_to_run $numtorun --finetune --log_save_dir $logdir

    python scripts/convert_validity_logs.py --log_dir $logdir --target_datasets $dataset \
            --filename $dataset"_validity.csv" --finetune
else 
    python sns/sns_cross_model.py $modelname $dataset "l1" --model_cnt $nummodels \
            --num_to_run $numtorun --cfx_dir $cfxdir --log_save_dir $logdir

    python scripts/convert_validity_logs.py --log_dir $logdir --target_datasets $dataset \
            --filename $dataset"_validity.csv" 
fi  

# step 3: compute delta-robustness 
logdir="logs"
for i in $(seq 0 $modelcnt)
do
    modelname=$dataset"Standard"$i
    if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
        python eval.py $modelname $dataset --save_dir $modeldir --num_to_run $numtorun \
            --cfx_save_dir $cfxdir --log_save_dir $logdir --skip_milp --cfx_technique "sns" --finetune
    else 
        python eval.py $modelname $dataset --save_dir $modeldir --num_to_run $numtorun \
            --cfx_save_dir $cfxdir --log_save_dir $logdir --cfx_technique "sns" --skip_milp
    fi
done

python scripts/convert_logs.py --log_dir $logdir --skip_milp --target_datasets $dataset --filename $dataset"_robustness.csv"
