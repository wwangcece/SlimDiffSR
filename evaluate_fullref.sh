HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
python -u evaluate.py \
--ref_path=/mnt/massive/wangce/LightSD/dataset/test-dota/images \
--in_path=result/Ours-final-8x/dota \
--out_path=logs/Final-8x/Ours_8x/dota

