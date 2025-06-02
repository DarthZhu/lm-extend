export CUDA_VISIBLE_DEVICES=0,1,2,3

BASEMODELNAME="Qwen2-7B-Instruct"
MODELNAME="data/local_llms/ft-300"
MODELNICKNAME=$(basename "$MODELNAME")

echo "Evaluating $MODELNICKNAME"

# cd data/models
# git lfs install
# git clone https://huggingface.co/$MODELNAME
# cd ../..

if [ "$BASEMODELNAME" == "$MODELNICKNAME" ]; then
    echo "Loading the base model..."
    MODELPATH="data/models/$MODELNICKNAME"
else
    echo "Saving $MODELNICKNAME..."
    # python src/utils/save_model_local.py \
    #     --base_model_name $BASEMODELNAME \
    #     --pretrained_model_name $MODELNICKNAME
    echo "Saving done"
    MODELPATH="data/local_llms/$MODELNICKNAME"
fi

echo "Start to inference"

echo "Inference mmlu answers..."
python src/inference.py \
    --dataset mmlu \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1

echo "Inference mmlu-pro answers..."
python src/inference.py \
    --dataset mmlu-pro \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1

echo "Inference ifeval answers..."
python src/inference.py \
    --dataset ifeval \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1

echo "Inference gpqa answers..."
python src/inference.py \
    --dataset gpqa \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1 \
    --ask_twice

echo "Inference math answers..."
python src/inference.py \
    --dataset math \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1 \
    --ask_twice

echo "Inference mmmlu answers..."
python src/inference.py \
    --dataset mmmlu \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1

echo "Inference humaneval answers..."
python src/inference.py \
    --dataset humaneval \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 200

echo "Inference harmbench..."
python src/inference.py \
    --dataset harmbench \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 100

echo "Inference passkey..."
python src/inference.py \
    --dataset passkey \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1

echo "Inference zeroscrolls..."
python src/inference.py \
    --dataset zeroscrolls \
    --pretrained_model_name $MODELPATH \
    --batch_size 16 \
    --num_sampling 1
