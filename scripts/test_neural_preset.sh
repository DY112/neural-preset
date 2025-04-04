CUDA_VISIBLE_DEVICES=1 python main.py \
model.name=neural_styler \
model.ver=v1 \
model.solver=v1 \
mode=test \
data.name=test_data \
load.ckpt_path=../ckpts/best.ckpt  # Update this path to match your downloaded checkpoint filename