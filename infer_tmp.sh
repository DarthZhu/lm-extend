export CUDA_VISIBLE_DEVICES=4,5,6,7

BASEMODELNAME="Qwen2-7B-Instruct"
MODELNAME="Qwen2-merged-avg"
MODELNICKNAME=$(basename "$MODELNAME")

if [ "$BASEMODELNAME" == "$MODELNICKNAME" ]; then
    echo "Loading the base model..."
    MODELPATH="data/models/$MODELNICKNAME"
else
    echo "Loading $MODELNICKNAME..."
    MODELPATH="data/local_llms/$MODELNICKNAME"
fi

echo "Evaluating $MODELNICKNAME"

# echo "Inference harmbench..."
# python src/inference.py \
#     --dataset harmbench \
#     --pretrained_model_name $MODELPATH \
#     --batch_size 16 \
#     --num_sampling 100

# echo "Inference passkey..."
# python src/inference.py \
#     --dataset passkey \
#     --pretrained_model_name $MODELPATH \
#     --batch_size 16 \
#     --num_sampling 1

# echo "Inference zeroscrolls..."
# python src/inference.py \
#     --dataset zeroscrolls \
#     --pretrained_model_name $MODELPATH \
#     --batch_size 16 \
#     --num_sampling 1

echo "Inference gpqa answers..."
python src/inference.py \
    --dataset gpqa \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1 \
    --ask_twice