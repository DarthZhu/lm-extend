# STEP="700"
# MODEL_PATH="output/checkpoints/debug/checkpoint-$STEP"
MODEL_PATH="output/checkpoints/video/checkpoint-156"
SAVE_MM_MODEL_PATH="data/models/ft-video"
SAVE_LLM_MODEL_PATH="data/local_llms/ft-video"

python $MODEL_PATH/zero_to_fp32.py $MODEL_PATH $SAVE_MM_MODEL_PATH --safe_serialization

cp data/models/Qwen2-VL-7B-Instruct/tokenizer.json $SAVE_MM_MODEL_PATH/tokenizer.json
cp $MODEL_PATH/added_tokens.json $SAVE_MM_MODEL_PATH/added_tokens.json
cp $MODEL_PATH/config.json $SAVE_MM_MODEL_PATH/config.json
cp $MODEL_PATH/generation_config.json $SAVE_MM_MODEL_PATH/generation_config.json
cp $MODEL_PATH/special_tokens_map.json $SAVE_MM_MODEL_PATH/special_tokens_map.json
cp $MODEL_PATH/tokenizer_config.json $SAVE_MM_MODEL_PATH/tokenizer_config.json
cp $MODEL_PATH/vocab.json $SAVE_MM_MODEL_PATH/vocab.json
cp data/models/Qwen2-VL-7B-Instruct/preprocessor_config.json $SAVE_MM_MODEL_PATH/preprocessor_config.json

python src/utils/save_model_local.py \
    --base_model_name Qwen2-7B-Instruct \
    --pretrained_model_name ft-video \
