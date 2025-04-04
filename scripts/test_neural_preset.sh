CUDA_VISIBLE_DEVICES=1 python main.py \
model.name=neural_styler \
model.ver=v1 \
model.solver=v1 \
mode=test \
data.name=test_data \
load.ckpt_path=../ckpts/250331_083655_neural_styler_v1/best_val_losses_total_loss_epoch=0031.ckpt \