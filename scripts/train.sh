# wachter
#CUDA_VISIBLE_DEVICES="" python train.py ours_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2 \
#  --model IBP --cfx wachter --epoch 100 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.2 --wandb \
#  >ours_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES="" python train.py ours_model_wacther_i100_l0_001_ls10_e100_c10_eps0_05_beps0_05_r0_2 \
#  --model IBP --cfx wachter --epoch 100 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.2 --wandb \
#  --cfx_generation_freq 10 \
#  >ours_model_wacther_i100_l0_001_ls10_e100_c10_eps0_05_beps0_05_r0_2.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES="" python train.py ibp_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2 \
#  --model IBP --cfx wachter --epoch 100 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.2 --tightness ibp --wandb \
#  >ibp_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES="" python train.py crownibp_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2 \
#  --model IBP --cfx wachter --epoch 100 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.2 --tightness crownibp --wandb \
#  >crownibp_model_wacther_i100_l0_001_ls10_e100_c20_eps0_05_beps0_05_r0_2.log 2>&1 &

# proto
#CUDA_VISIBLE_DEVICES="" python train.py standard_model_proto_onehot_e50 --model Standard --cfx proto --onehot \
#  --epoch 50 --wandb 2>&1 &
#CUDA_VISIBLE_DEVICES="" python train.py ours_model_proto_onehot_t100_e50_c20_eps0_05_beps0_05_r0_4 --model IBP \
#  --cfx proto --onehot --epoch 50 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.4 --wandb 2>&1 &
#CUDA_VISIBLE_DEVICES="" python train.py ibp_model_proto_onehot_t100_e50_c20_eps0_05_beps0_05_r0_4 --model IBP \
#  --cfx proto --onehot --epoch 50 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.4 --tightness ibp --wandb 2>&1 &
#CUDA_VISIBLE_DEVICES="" python train.py crown_ibp_model_proto_onehot_t100_e50_c20_eps0_05_beps0_05_r0_4 --model IBP \
#  --cfx proto --onehot --epoch 50 --epsilon 5e-2 --bias_epsilon 5e-2 --ratio 0.4 --tightness crownibp --wandb 2>&1 &

#CUDA_VISIBLE_DEVICES="" python train.py heloc_IBP_model_counternet_ibp --model IBP --cfx counternet --epoch 100 \
#--epsilon 5e-2 --bias_epsilon 5e-2 --tightness ibp --wandb --config assets/heloc.json --cfx_generation_freq 1 \
#--ratio 0.2 2>&1
#CUDA_VISIBLE_DEVICES="" python train.py heloc_IBP_model_counternet_crownibp --model IBP --cfx counternet --epoch 100 \
#--epsilon 5e-2 --bias_epsilon 5e-2 --tightness crownibp --wandb --config assets/heloc.json --cfx_generation_freq 1 \
#--ratio 0.2 2>&1

CUDA_VISIBLE_DEVICES="" python train.py ctg_IBP_model_counternet_ibp_r0_4 --model IBP --cfx counternet --epoch 200 \
--epsilon 5e-2 --bias_epsilon 5e-2 --tightness ibp --wandb --config assets/ctg.json --cfx_generation_freq 1 \
--ratio 0.4 >/dev/null 2>&1
CUDA_VISIBLE_DEVICES="" python train.py ctg_IBP_model_counternet_ibp_r0_8 --model IBP --cfx counternet --epoch 200 \
--epsilon 5e-2 --bias_epsilon 5e-2 --tightness ibp --wandb --config assets/ctg.json --cfx_generation_freq 1 \
--ratio 0.8 >/dev/null 2>&1
CUDA_VISIBLE_DEVICES="" python train.py ctg_IBP_model_counternet_crownibp_r0_4 --model IBP --cfx counternet --epoch 200 \
--epsilon 5e-2 --bias_epsilon 5e-2 --tightness crownibp --wandb --config assets/ctg.json --cfx_generation_freq 1 \
--ratio 0.4 >/dev/null 2>&1
CUDA_VISIBLE_DEVICES="" python train.py ctg_IBP_model_counternet_crownibp_r0_8 --model IBP --cfx counternet --epoch 200 \
--epsilon 5e-2 --bias_epsilon 5e-2 --tightness crownibp --wandb --config assets/ctg.json --cfx_generation_freq 1 \
--ratio 0.8 >/dev/null 2>&1
#CUDA_VISIBLE_DEVICES="" python train.py ctg_IBP_model_counternet_ibp --model IBP --cfx counternet --epoch 200 \
#--epsilon 5e-2 --bias_epsilon 5e-2 --tightness ibp --wandb --config assets/ctg.json --cfx_generation_freq 1 \
#--ratio 0.2 >/dev/null 2>&1
#CUDA_VISIBLE_DEVICES="" python train.py ctg_IBP_model_counternet_crownibp --model IBP --cfx counternet --epoch 200 \
#--epsilon 5e-2 --bias_epsilon 5e-2 --tightness crownibp --wandb --config assets/ctg.json --cfx_generation_freq 1 \
#--ratio 0.2 >/dev/null 2>&1