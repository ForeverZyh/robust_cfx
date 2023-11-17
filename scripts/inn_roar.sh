# run inn_exp.py ten times with different seeds 0-9

dataset=(heloc ctg)
ps=("0.0992" "0.2396")
ss=("0.8207" "0.7730")
for ((i = 0; i < ${#dataset[@]}; i++)); do
  d=${dataset[$i]}
  p=${ps[$i]}
  s=${ss[$i]}
  for seed in {0..9}; do
          python scripts/inn_exp.py $d IBP roar --cfx counternetours --generate_only --target_idxs $seed --force_regen_cfx \
            --target_p $p --target_s $s &
          python scripts/inn_exp.py $d Standard roar --cfx counternet --generate_only --target_idxs $seed --force_regen_cfx \
            --target_p $p --target_s $s &
#    CUDA_VISIBLE_DEVICES="" python eval.py Standard"$d"counternet"$seed" --cfx counternet --log_name \
#      Standard"$d"counternetroar"$seed".log --config assets/"$d".json \
#      --seed $seed >/dev/null 2>&1 &
#    CUDA_VISIBLE_DEVICES="" python eval.py IBP"$d"counternetours"$seed" --cfx counternet --log_name \
#      IBP"$d"counternetoursroar"$seed".log --config assets/"$d".json \
#      --seed $seed >/dev/null 2>&1 &
  done
  wait
  #  python scripts/inn_exp.py $d IBP roar --cfx counternetours
  #  python scripts/inn_exp.py $d Standard roar --cfx counternet
done
