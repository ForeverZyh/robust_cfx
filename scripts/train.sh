# Train 10 models (or 10 sets of original/shifted models) using VeriTraCER (IBP)

dataset=$1

if [ -z "$dataset" ]; then
    echo "Please provide a dataset"
    exit 1
fi

modeldir="trained_models"
epochs=100

for i in {0..9} 
do
    modelname=$dataset"IBP"$i 

    python train.py $modelname $dataset --save_dir $modeldir --model IBP --epoch $epochs --seed $i

    # if dataset==who or dataset==ctg need to finetune
    if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
        epochs=20
        python train.py $modelname $dataset --save_dir $modeldir --model IBP --epoch $epochs --finetune --seed $i
        epochs=100
    fi
done


# Uncomment to run LOO experiments
# for i in {0..9} 
# do
#     modelname=$dataset"IBP_LOO"$i 

#     if [ $dataset == "who" ] || [ $dataset == "ctg" ]; then
#         exit 1
#     fi

#     python train.py $modelname $dataset --save_dir $modeldir --model IBP --epoch $epochs --seed 0 --remove_pct 1 --removal_start $i

# done