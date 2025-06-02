import os
import json
import torch
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
# from accelerate import Accelerator

# from src.prompts import qa_prompt
# from src.utils.load_weights import load_weights
from src.utils.load_dataset import get_datas


def main():
    # set args
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--pretrained_model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--ask_twice", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_sampling", type=int, default=1)
    
    args = parser.parse_args()
    
    # initialize dataset, model, and tokenizer
    # model, tokenizer = load_weights(args.base_model_name, args.pretrained_model_name)
    # datas, prompt = get_datas(args.dataset)
    # model = model.to("cuda")
    
    model = LLM(
        model=args.pretrained_model_name,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        # max_seq_len_to_capture=200000,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, trust_remote_code=True)
    tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if args.num_sampling > 1:
        temperature = 1.0
    else:
        temperature = 0.0
    sampling_params = SamplingParams(
        n=args.num_sampling,
        temperature=temperature,
        max_tokens=4096,
        # logprobs=20,
    )
    datas, prompt = get_datas(args.dataset)
    
    # set output path
    if args.pretrained_model_name is None:
        model_name = args.base_model_name.split("/")[-1]
    else:
        model_name = args.pretrained_model_name.split("/")[-1]
    output_dir = os.path.join(args.output_dir, model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, f"{args.dataset}.txt")
    
    # prepare batch data
    all_texts = []
    for data in datas:
        data_id = data["id"]
        if "llava-1.5" in model_name:
            # this is for llava, need to improve
            if prompt:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [

                            {"type": "text", "text": data["text"] + "\n"},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [

                            {"type": "text", "text": data["text"] + "\n"},
                        ],
                    }
                ]
        else:
            if prompt:
                messages = [
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": data["text"] + "\n",
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": data["text"] + "\n",
                    }
                ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_texts.append({data_id: text})
    
    number_batch = int(len(all_texts) / args.batch_size) + 1
    batch_texts = []
    for i in range(number_batch):
        if i < number_batch - 1:
            batch_texts.append(all_texts[i * args.batch_size: (i + 1) * args.batch_size])
        else:
            batch_texts.append(all_texts[i * args.batch_size:])
    
    # inference
    progress_bar = tqdm(range(number_batch))
    for batch in batch_texts:
        batch_ids = [list(b.keys())[0] for b in batch]
        batch_texts = [list(b.values())[0] for b in batch]
        outputs = model.generate(
            batch_texts,
            sampling_params=sampling_params,
        )
        for data_id, output in zip(batch_ids, outputs):
            if len(output.outputs) == 1:
                generated_texts = output.outputs[0].text
            else:
                generated_texts = [o.text for o in output.outputs]
            with open(output_path, "a+") as fout:
                fout.write(f"{json.dumps({data_id: generated_texts})}\n")
        progress_bar.update(1)
    
    
    
    if args.ask_twice:
        id2answer = {}
        with open(output_path, "r") as fin:
            for line in fin.readlines():
                id2answer.update(json.loads(line))
        # prepare batch data
        all_texts = []
        for data in datas:
            data_id = data["id"]
            first_prediction = id2answer.get(data_id)
            if "llava" in model_name:
                # this is for llava, need to improve
                if prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [

                                {"type": "text", "text": data["text"] + "\n"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [

                                {"type": "text", "text": first_prediction + "\n"},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [

                                {"type": "text", "text": "The final answer is:"},
                            ],
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [

                                {"type": "text", "text": data["text"] + "\n"},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [

                                {"type": "text", "text": first_prediction + "\n"},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [

                                {"type": "text", "text": "The final answer is:"},
                            ],
                        }
                    ]
            else:
                if prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": data["text"] + "\n",
                        },
                        {
                            "role": "assistant",
                            "content": first_prediction + "\n",
                        },
                        {
                            "role": "user",
                            "content": "The final answer is:",
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": data["text"] + "\n",
                        },
                        {
                            "role": "assistant",
                            "content": first_prediction + "\n",
                        },
                        {
                            "role": "user",
                            "content": "The final answer is:",
                        }
                    ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append({data_id: text})
        
        number_batch = int(len(all_texts) / args.batch_size) + 1
        batch_texts = []
        for i in range(number_batch):
            if i < number_batch - 1:
                batch_texts.append(all_texts[i * args.batch_size: (i + 1) * args.batch_size])
            else:
                batch_texts.append(all_texts[i * args.batch_size:])
        
        # inference
        progress_bar = tqdm(range(number_batch))
        for batch in batch_texts:
            batch_ids = [list(b.keys())[0] for b in batch]
            batch_texts = [list(b.values())[0] for b in batch]
            outputs = model.generate(
                batch_texts,
                sampling_params=sampling_params,
            )
            for data_id, output in zip(batch_ids, outputs):
                if len(output.outputs) == 1:
                    generated_texts = output.outputs[0].text
                else:
                    generated_texts = [o.text for o in output.outputs]
                with open(output_path + ".twice", "a+") as fout:
                    fout.write(f"{json.dumps({data_id: generated_texts})}\n")
            progress_bar.update(1)
        
        
            
if __name__ == "__main__":
    main()