import os
import json
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from argparse import ArgumentParser

import tiktoken
from openai import BadRequestError
from typing import List, Dict, AnyStr
from tenacity import stop_after_attempt, retry, wait_random_exponential, retry_if_not_exception_type
from multiprocessing import Pool, Value

from src.utils.load_dataset import get_datas

with open("data/secrets/openai", "r") as fin:
    key = fin.readlines()[0]
    os.environ["OPENAI_API_KEY"] = key

class EvalOutput(BaseModel):
    rationale: str
    is_correct: bool

def read_pred_file(input_file):
    preds = {}
    with open(input_file, "r") as fin:
        for line in fin.readlines():
            preds.update(json.loads(line))
    return preds

@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60), retry=retry_if_not_exception_type(BadRequestError))
def evaluate_single(
    inputs,
):
    pred_id, gold_answer, prediction = inputs
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Determine whether the prediction is correct given the gold answer. Remember the gold answer is always correct."},
            {"role": "user", "content": f"Gold Answer: {gold_answer}\nPrediction:{prediction}"}
        ],
        response_format=EvalOutput
    )
    output = completion.choices[0].message.parsed
    return (pred_id, output)

def evaluate(gold_answer, prediction):
    """For debug; single thread, cost too much time"""
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Determine whether the prediction is correct given the gold answer. Remember the gold answer is always correct."},
            {"role": "user", "content": f"Gold Answer: {gold_answer}\nPrediction:{prediction}"}
        ],
        response_format=EvalOutput
    )
    output = completion.choices[0].message.parsed
    print(gold_answer)
    print(prediction)
    print(output)
    input()
    return output

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--compared_file", type=str, default=None)
    parser.add_argument("--cutoff", action="store_true")
    
    args = parser.parse_args()
    
    # load datas and preds
    datas, _ = get_datas(args.dataset)
    preds = read_pred_file(args.file)

    # prepare batch inputs
    batch_inputs = []
    for data in datas:
        _ = data["id"]
        pred = preds.get(f"{_}")
        if pred is None:
            continue
        if args.cutoff:
            pred = pred.split("\n")[-1]
        answer = data["answer"]
        batch_inputs.append([_, answer, pred])
        
        # uncomment here to debug
        # evaluate(answer, pred)
        
    cnt_valid = 0
    cnt_correct = 0
    with Pool(32) as p:
        batch_outputs = list(tqdm(p.imap(evaluate_single, batch_inputs), total=len(batch_inputs)))
    
    for output in batch_outputs:
        _, evaluation = output
        if evaluation.is_correct:
            cnt_correct += 1
        cnt_valid += 1
        with open(f"{args.file}.eval", "a+") as fout:
            fout.write(f"{json.dumps({_: evaluation.model_dump()})}\n")
    print(f"All sample number: {cnt_valid}")
    print(f"Correct sample number: {cnt_correct}")
    print(f"Accuracy: {cnt_correct / cnt_valid}")
    
if __name__ == "__main__":
    main()