import json
from datasets import load_dataset, concatenate_datasets
from src.prompts import qa_prompt, context_prompt

question_with_choices = "Question: {question}\nChoices:\n{choices}"
question_with_choices_and_context = "{context}\n\nQuestion: {question}\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\n\n{choices}"

def format_choices(choices):
    choices_text = ""
    for _, choice in enumerate(choices):
        choice_label = chr(ord("A") + _)
        choices_text += f"{choice_label}. {choice.strip()}\n"
    return choices_text.strip()

def get_datas(dataset_name):
    """
    Convert the original datasets into evaluation form:
    {
        "text": text,
        "answer": answer,
    }
    """
    datas = []
    prompt = None
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "mmlu":
        dataset = load_dataset("data/datasets/mmlu", "all")["test"]
        prompt = qa_prompt
        for _, data in enumerate(dataset):
            question = data["question"]
            choices = data["choices"]
            choices_text = format_choices(choices)
            text = question_with_choices.format(question=question, choices=choices_text)
            answer = data["answer"]
            datas.append({
                "id": _,
                "text": text,
                "answer": answer,
            })
    elif dataset_name == "mmlu-pro":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")['test']
        prompt = qa_prompt
        for data in dataset:
            question = data["question"]
            choices = data["options"]
            data_id =  data["question_id"]
            choices_text = format_choices(choices)
            text = question_with_choices.format(question=question, choices=choices_text)
            answer = data["answer"]
            datas.append({
                "id": data_id,
                "text": text,
                "answer": answer,
            })
    elif dataset_name == "gpqa":
        import random
        random.seed(42)
        dataset = load_dataset("data/datasets/gpqa", "gpqa_diamond")['train']
        for _, data in enumerate(dataset):
            question = data["Question"]
            text = f"What is the correct answer to this question: {question}\n\n"
            choices = [data["Correct Answer"], data["Incorrect Answer 1"],data["Incorrect Answer 2"],data["Incorrect Answer 3"]]
            random.shuffle(choices)
            answer = choices.index(data["Correct Answer"]) + 1
            choices_text = format_choices(choices)
            text += choices_text
            text += f"\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""            
            datas.append({
                "id": _,
                "text": question,
                "answer": answer,
                "choices": choices,
            })
    elif dataset_name=='mmmlu':
        dataset = load_dataset("openai/MMMLU", "default")['test']
        prompt = qa_prompt
        for _, data in enumerate(dataset):
            question = data["Question"]
            choices = []
            for i in range(ord('A'), ord('D') + 1): 
                choices.append(data[f"{chr(i)}"])
            choices_text = format_choices(choices)
            text = question_with_choices.format(question=question, choices=choices_text)
            answer = data["Answer"]
            datas.append({
                "id": _,
                "text": text,
                "answer": answer,
                "choices": choices,
            })
    elif dataset_name=='math': # 非选择题,需要LLM judge一下
        dataset = load_dataset("lighteval/MATH", "all")['test']
        prompt = "Let's think step by step.\n"
        for _, data in enumerate(dataset):
            question = data["problem"]
            answer = data["solution"]
            datas.append({
                "id": _,
                "text": question,
                "answer": answer,
            })
    elif dataset_name == "humaneval":# 非选择题,需要LLM judge一下
        dataset = load_dataset("evalplus/humanevalplus")['test']
        prompt = "Please complete the task and output the code in the markdown format (```python\nyour code\n```)."
        for _, data in enumerate(dataset):
            question = data["prompt"]
            answer = data["canonical_solution"]
            datas.append({
                "id": data['task_id'],
                "text": question,
                "answer": answer,
            })

    elif dataset_name == "ifeval":
        dataset = load_dataset("data/datasets/IFEval")["train"]
        prompt = None
        for data in dataset:
            data_id = data["key"]
            text = data["prompt"]
            datas.append({
                "id": data_id,
                "text": text,
            })
    elif dataset_name == "infinitebench":
        from datasets import Value, Sequence, Features
        ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"), "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
        dataset = load_dataset("data/datasets/InfiniteBench", features=ft)["longbook_choice_eng"]
        prompt = context_prompt
        for data in dataset:
            data_id = data["id"]
            context = data["context"]
            question = data["input"]
            choices = data["options"]
            choices_text = format_choices(choices)
            answer = choices.index(data["answer"][0])
            text = question_with_choices_and_context.format(context=context, question=question, choices=choices_text)
            datas.append({
                "id": data_id,
                "text": text,
                "answer": answer,
            })
    elif dataset_name == "harmbench":
        dataset = load_dataset("data/datasets/HarmBench", "standard")["train"]
        prompt = None
        for _, data in enumerate(dataset):
            data_id = _
            text = data["prompt"]
            datas.append({
                "id": data_id,
                "text": text,
            })
    elif dataset_name == "passkey":
        with open("data/datasets/passkey/data.json", "r") as fin:
            dataset = json.load(fin)
        prompt = None
        for _, data in enumerate(dataset):
            data_id = _
            prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
            text = data["question"]
            answer = data["answer"]
            datas.append({
                "id": data_id,
                "text": text,
                "answer": answer,
            })
    elif dataset_name == "zeroscrolls":
        dataset = load_dataset("tau/zero_scrolls", "quality", trust_remote_code=True)["validation"]
        # prompt = qa_prompt
        for _, data in enumerate(dataset):
            data_id = data["id"]
            text = data["input"]
            datas.append({
                "id": data_id,
                "text": text,
                "answer": data["output"]
            })
    
    print(f"Dataset length: {len(datas)}")
    return datas, prompt

def get_multimodal_datas(dataset_name, is_eval=False):
    """
    Convert the original datasets into evaluation form:
    {
        "text": text,
        "answer": answer,
    }
    """
    datas = []
    prompt = None
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "mmmu":
        import ast
        available_configs = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
        # Load the test split for all subsets
        test_subsets = {config: load_dataset("MMMU/MMMU", config, split="validation") for config in available_configs}
        prompt = qa_prompt
        
        datas = []
        idx = 0
        for subset, subset_datas in test_subsets.items():
            for data in subset_datas:
                question = data["question"].replace("<image 1>", "").strip()
                choices = ast.literal_eval(data["options"])
                choices_text = format_choices(choices)
                text = question_with_choices.format(question=question, choices=choices_text)
                answer = data["answer"]
                if is_eval:
                    datas.append({
                        "id": idx,
                        "text": text,
                        "answer": answer,
                        "subset": subset,
                    })
                    idx += 1
                else:
                    datas.append({
                        "id": idx,
                        "text": text,
                        "answer": answer,
                        "image": data["image_1"].convert("RGB").resize((224, 224)),
                        "subset": subset,
                    })
                    idx += 1
        return datas, prompt
    elif dataset_name == "video-mme":
        import os
        dataset = load_dataset("lmms-lab/Video-MME")["test"]
        prompt = qa_prompt
        video_dir = "/home/darthzhu/.cache/huggingface/videomme/data"
        for _, data in enumerate(dataset):
            data_id = data["question_id"]
            question = data["question"]
            choices = [option.split(".")[1].strip() for option in data["options"]]
            choices_text = format_choices(choices)
            text = question_with_choices.format(question=question, choices=choices_text)
            video_id = data["videoID"]
            datas.append({
                "id": data_id,
                "text": text,
                "answer": data["answer"],
                "video": os.path.join(video_dir, f"{video_id}.mp4") 
            })
        return datas, prompt

if __name__ == "__main__":
    datas, prompt = get_multimodal_datas("video-mme")
    for data in datas:
        print(data)
        input()