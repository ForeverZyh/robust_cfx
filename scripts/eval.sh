wachter_models=(
  'ours_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2'
  'ours_model_wacther_i100_l0_001_ls10_e100_c10_eps0_05_beps0_05_r0_2'
  'ibp_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2'
  'crownibp_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2'
  'ours_model_wachter'
)

wachter_max_iters=(200)
wachter_lam_inits=(1e-2 1e-3 1e-4)
wachter_max_lam_steps=(10)

proto_models=(
)

proto_thetas=(1 10 100 1000)

EPS=1e-2
BEPS=1e-3

mkdir logs

for model in "${proto_models[@]}"; do
  for proto_theta in "${proto_thetas[@]}"; do
    CUDA_VISIBLE_DEVICES="" python eval.py $model --cfx proto --epsilon $EPS --bias_epsilon $BEPS \
      --proto_theta $proto_theta --onehot --log_name eval_${model}_t${proto_theta}.log >/dev/null 2>&1 &
  done
done

for model in "${wachter_models[@]}"; do
  for wachter_max_iter in "${wachter_max_iters[@]}"; do
    for wachter_lam_init in "${wachter_lam_inits[@]}"; do
      for wachter_max_lam_step in "${wachter_max_lam_steps[@]}"; do
        CUDA_VISIBLE_DEVICES="" python eval.py $model --cfx wachter --epsilon $EPS --bias_epsilon $BEPS \
          --wachter_max_iter $wachter_max_iter --wachter_lam_init $wachter_lam_init \
          --wachter_max_lam_steps $wachter_max_lam_step \
          --log_name eval_${model}_i${wachter_max_iter}_l${wachter_lam_init}_ls${wachter_max_lam_step}.log >/dev/null 2>&1 &
      done
    done
  done
done
