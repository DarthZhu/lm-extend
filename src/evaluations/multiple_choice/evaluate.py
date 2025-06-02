import re
import json
from argparse import ArgumentParser

from src.utils.load_dataset import get_datas, get_multimodal_datas

def read_pred_file(input_file):
    preds = {}
    with open(input_file, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds

def clean_answer(answer):
    pattern = r"\b[A-J]\b|[A-J](?=\s|:|.)"
    match = re.search(pattern, answer)
    if match is None:
        return None
    return match.group()

def match_answer(choices, gold_answer, prediction):
    prediction = prediction.split("\n")[-1]
    try:
        prediction = json.loads(prediction.strip())["answer"]
    except:
        pass
    gold_choice = choices[gold_answer - 1]
    if gold_choice in prediction or prediction in gold_choice:
        return True
    # print(gold_choice)
    # print(prediction)
    # input()
    return False
    

# def evaluate(choices, gold_answer, predict_answer):
def evaluate(gold_answer, predict_answer):
    if type(gold_answer) is int:
        gold_answer_choice = chr(ord("A") + gold_answer)
    else:
        gold_answer_choice = gold_answer
    cleaned_predict_answer = clean_answer(predict_answer)
    if cleaned_predict_answer is None:
        print(predict_answer)
        # return match_answer(choices, gold_answer, predict_answer)
        return False
    if cleaned_predict_answer == gold_answer_choice:
        return True
    return False

def compare(answer_1, answer_2):
    cleaned_answer_1 = clean_answer(answer_1)
    cleaned_answer_2 = clean_answer(answer_2)
    if cleaned_answer_1 is None or cleaned_answer_2 is None:
        return False
    if cleaned_answer_1 == cleaned_answer_2:
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
    if len(datas) == 0:
        datas, _ = get_multimodal_datas(args.dataset, is_eval=True)
    preds = read_pred_file(args.file)
    
    if args.compared_file:
        compared_preds = read_pred_file(args.compared_file)
        
        cnt_valid = 0
        cnt_consistent = 0
        for _, data in enumerate(datas):
            pred_1 = preds.get(f"{_}")
            pred_2 = compared_preds.get(f"{_}")
            if pred_1 is None or pred_2 is None:
                continue
            if compare(pred_1):
                cnt_consistent += 1
            cnt_valid += 1
        print(f"All sample number: {cnt_valid}")
        print(f"Consistent sample number: {cnt_consistent}")
        print(f"Accuracy: {cnt_consistent / cnt_valid}")
            
    else:
        cnt_valid = 0
        cnt_correct = 0
        for data in datas:
            _ = data["id"]
            # choices = data["choices"]
            pred = preds.get(f"{_}")
            if pred is None:
                continue
            answer = data["answer"]
            # if evaluate(choices, answer, pred):
            if evaluate(answer, pred):
                cnt_correct += 1
            cnt_valid += 1
        print(f"All sample number: {cnt_valid}")
        print(f"Correct sample number: {cnt_correct}")
        print(f"Accuracy: {cnt_correct / cnt_valid}")
        
if __name__ == "__main__":
    main()