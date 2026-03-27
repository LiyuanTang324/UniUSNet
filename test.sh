export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12345 \
    omni_test.py \
    --root_path=../../data \
    --output_dir=exp_out/dua_20260326_sd42 \
    --cfg=configs/swin_tiny_patch4_window8_256_lite.yaml \
    --max_epochs=200 \
    --batch_size=32 \
    --base_lr=3e-4 \
    --seed=42 \
    --prompt

