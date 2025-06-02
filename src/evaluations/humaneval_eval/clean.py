import re
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--file", type=str, default="output/llava-1.5-7b-hf/humaneval.txt")
args = parser.parse_args()

id2solutions = {}
with open(args.file, "r") as fin:
    for line in fin.readlines():
        id2solutions.update(json.loads(line))

pattern = r"```python(.*?)```" 

target_path = "/".join(args.file.split("/")[:-1]) + "/humaneval.jsonl"

if os.path.exists(target_path):
    os.remove(target_path)

for data_id in tqdm(list(id2solutions.keys())):
    solutions = id2solutions.get(data_id)
    for s in solutions:
        matches = re.findall(pattern, s, re.DOTALL)
        if len(matches) > 0:
            with open(target_path, "a+") as fout:
                fout.write(json.dumps({
                    "task_id": data_id,
                    "solution": matches[0],
                }) + "\n")
        else:
            with open(target_path, "a+") as fout:
                fout.write(json.dumps({
                    "task_id": data_id,
                    "solution": "",
                }) + "\n")