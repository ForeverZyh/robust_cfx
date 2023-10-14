MODEL_CNT=3

datasets=(
  'heloc'
)

epochs=(
  100
)

len=${#datasets[@]}

ibp_approaches=(
  'ibp'
  'crownibp'
#  'ours'
)

for ((i = 0; i < MODEL_CNT; i++)); do
  for ((j = 0; j < len; j++)); do
    dataset=${datasets[$j]}
    epoch=${epochs[$j]}
#    CUDA_VISIBLE_DEVICES="" python train.py Standard"$dataset"counternet"$i" --model Standard --cfx counternet \
#      --epoch $epoch --config assets/"$dataset".json --seed $i --wandb 2>&1
#    rm saved_cfxs/Standard"$dataset"counternet"$i" -f
#    CUDA_VISIBLE_DEVICES="" python eval.py Standard"$dataset"counternet"$i" --cfx counternet --log_name \
#      Standard"$dataset"counternet"$i".log --epsilon $EPS --bias_epsilon $BEPS --config assets/"$dataset".json \
#      --seed $i 2>&1
    for ibp_approach in "${ibp_approaches[@]}"; do
      CUDA_VISIBLE_DEVICES="" python train.py IBP"$dataset"counternet"$ibp_approach""$i" --model IBP --cfx counternet \
        --epoch $epoch --tightness $ibp_approach --config assets/"$dataset".json \
        --seed $i --cfx_generation_freq 1 --ratio 0.05 --wandb 2>&1
      rm saved_cfxs/IBP"$dataset"counternet"$ibp_approach""$i" -f
      CUDA_VISIBLE_DEVICES="" python eval.py IBP"$dataset"counternet"$ibp_approach""$i" --cfx counternet --log_name \
        IBP"$dataset"counternet"$ibp_approach""$i".log --config assets/"$dataset".json \
        --seed $i >/dev/null 2>&1 &
    done
  done
done
