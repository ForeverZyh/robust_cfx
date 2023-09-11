# wachter
#CUDA_VISIBLE_DEVICES="" python train.py standard_model_wacther_e50 --model Standard --cfx wachter --epoch 50
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e50_c20_eps0_01_beps0_001 --model IBP --cfx wachter --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-3
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e50_c20_inc_eps0_01_beps0_001 --model IBP --cfx wachter --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-3 --inc_regenerate

#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001 \
#  --model IBP --cfx wachter --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 \
#  >IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001.log 2>&1 &
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e50_c10_eps0_01_beps0_001 \
#  --model IBP --cfx wachter --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-3 --cfx_generation_freq 10 \
#  >IBP_model_wacther_i100_l0_001_ls10_e50_c10_eps0_01_beps0_001.log 2>&1 &

# CUDA_VISIBLE_DEVICES="" python train.py standard_model_wacther_e100 --model Standard --cfx wachter --epoch 100 2>&1 &
# CUDA_VISIBLE_DEVICES="" python train.py standard_model_wacther_e200 --model Standard --cfx wachter --epoch 200 2>&1 &
# CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e200_c20_eps0_01_beps0_001 \
#   --model IBP --cfx wachter --epoch 200 --epsilon 1e-2 --bias_epsilon 1e-3 \
#   >IBP_model_wacther_i100_l0_001_ls10_e200_c20_eps0_01_beps0_001.log 2>&1 &
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e100_c20_inc_eps0_01_beps0_001_r0_2 \
#  --model IBP --cfx wachter --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --ratio 0.2 --inc_regenerate 2>&1 &

# CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001_r0_2 \
#   --model IBP --cfx wachter --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --ratio 0.2 \
#   >IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001_r0_2.log 2>&1 &

# CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001_r0_4 \
#   --model IBP --cfx wachter --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --ratio 0.4 \
#   >IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001_r0_4.log 2>&1 &

# CUDA_VISIBLE_DEVICES="" python train.py IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001_r0_8 \
#   --model IBP --cfx wachter --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --ratio 0.8 \
#   >IBP_model_wacther_i100_l0_001_ls10_e100_c20_eps0_01_beps0_001_r0_8.log 2>&1 &

# proto
#CUDA_VISIBLE_DEVICES="" python train.py standard_model_proto_onehot_e50 --model Standard --cfx proto --onehot --epoch 50
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_proto_onehot_t100_e50_c20_eps0_01_beps0_001 --model IBP --cfx proto --onehot --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-3
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_proto_onehot_t100_e50_c20_inc_eps0_01_beps0_001 --model IBP --cfx proto --onehot --epoch 50 --epsilon 1e-2 --bias_epsilon 1e-3 --inc_regenerate


CUDA_VISIBLE_DEVICES="" python train.py standard_model_proto_onehot_e100 --model Standard --cfx proto --onehot --epoch 100
CUDA_VISIBLE_DEVICES="" python train.py IBP_model_proto_onehot_t100_e100_c20_eps0_01_beps0_001 --model IBP --cfx proto --onehot --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_proto_onehot_t100_e100_c20_eps0_01_beps0_001_r0_2 --model IBP --cfx proto --onehot --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --ratio 0.2
#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_proto_onehot_t100_e100_c20_eps0_01_beps0_001_r0_4 --model IBP --cfx proto --onehot --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --ratio 0.4


#CUDA_VISIBLE_DEVICES="" python train.py IBP_model_proto_onehot_t100_e100_c20_inc_eps0_01_beps0_001 --model IBP --cfx proto --onehot --epoch 100 --epsilon 1e-2 --bias_epsilon 1e-3 --inc_regenerate
