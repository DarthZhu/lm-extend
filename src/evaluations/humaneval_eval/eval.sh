MODELNAME="Qwen2-merged-weighted-all"

conda activate evalplus

python src/evaluations/humaneval_eval/clean.py --file output/$MODELNAME/humaneval.txt
evalplus.evaluate --samples output/$MODELNAME/humaneval.jsonl \
                  --dataset humaneval