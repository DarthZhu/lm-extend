accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=data/models/Qwen2-VL-7B-Instruct \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix Qwen2-VL-7B-Instruct \
    --output_path ./output/