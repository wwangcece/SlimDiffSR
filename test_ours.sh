HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
python -u test_ours.py \
--upscale=4 \
--LR_dir=path/to/lr_img \
--SR_dir=result/SR \
--model_dir=weight/net_params_4x.pkl