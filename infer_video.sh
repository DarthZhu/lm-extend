export CUDA_VISIBLE_DEVICES=0,1,2,3

MODELNAME="Qwen2-VL-7B-Instruct-Qwen2-merged-avg-text-VL-video-onevision"

echo "Evaluating $MODELNAME"

# cd data/models
# git lfs install
# git clone https://huggingface.co/$MODELNAME
# cd ../..

# if [ "$BASEMODELNAME" == "$MODELNICKNAME" ]; then
echo "Loading the base model..."
MODELPATH="data/local_llms/$MODELNAME"
# else
#     echo "Saving $MODELNICKNAME..."
#     # python src/utils/save_model_local.py \
#     #     --base_model_name $BASEMODELNAME \
#     #     --pretrained_model_name $MODELNICKNAME
#     echo "Saving done"
#     MODELPATH="data/local_llms/$MODELNICKNAME"
# fi

echo "Start to inference"

echo "Inference video-mme answers..."
python src/inference_video.py \
    --dataset video-mme \
    --pretrained_model_name $MODELPATH \
    --batch_size 4 \
    --num_sampling 1