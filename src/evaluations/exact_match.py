import re
import json
from argparse import ArgumentParser

from src.utils.load_dataset import get_datas

def read_pred_file(input_file):
    preds = {}
    with open(input_file, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds    

def evaluate(gold_answer, predict_answer):
# def evaluate(gold_answer, predict_answer):
    if gold_answer in predict_answer:
        return True
    return False

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--compared_file", type=str, default=None)
    
    args = parser.parse_args()
    
    # load datas and preds
    datas, _ = get_datas(args.dataset)
    preds = read_pred_file(args.file)
    
    cnt_valid = 0
    cnt_correct = 0
    for data in datas:
        _ = data["id"]
        pred = preds.get(f"{_}")
        if pred is None:
            continue
        answer = data["answer"]
        if evaluate(answer, pred):
        # if evaluate(answer, pred):
            cnt_correct += 1
        cnt_valid += 1
    print(f"All sample number: {cnt_valid}")
    print(f"Correct sample number: {cnt_correct}")
    print(f"Accuracy: {cnt_correct / cnt_valid}")
        
if __name__ == "__main__":
    main()