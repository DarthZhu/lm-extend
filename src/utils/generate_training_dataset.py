import os
import json
import random
from datasets import load_dataset

def generate_training_dataset():
    text_dataset = []
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    dataset = [data for data in dataset]
    random.shuffle(dataset)
    for data in dataset[:200000]:
        conversations = [
            {
                "from": "human",
                "value": data["query"],
            },
            {
                "from": "gpt",
                "value": data["response"],
            },
        ]
        text_dataset.append(
            {
                "conversations": conversations,
            }
        )
    
    image_dir = "data/datasets/training/VisualWebInstruct/imgs"
    image_dataset = []
    dataset = load_dataset("TIGER-Lab/VisualWebInstruct", "conversation")["train"]
    dataset = [data for data in dataset if data["image"] is not None]
    random.shuffle(dataset)
    for data in dataset:
        if len(image_dataset) >= 135000:
            break
        images = []
        is_complete = True
        for img in data["image"]:
            image_path = os.path.join(image_dir, img)
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                is_complete = False
                break
            images.append(image_path)
        if not is_complete:
            continue
        image_dataset.append(
            {
                "images": images,
                "conversations": data["conversations"],
            }
        )
        
    video_dir = "data/datasets/training/LLaVA-Video-178K"
    video_dataset = []
    dataset = []
    for split in load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1").keys():
        dataset.extend([data for data in load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1")[split]])
    random.shuffle(dataset)
    for data in dataset:
        if len(video_dataset) >= 65000:
            break
        video_path = os.path.join(video_dir, data["data_source"])
        video_path = os.path.join(video_path, data["video"])
        if os.path.exists(video_path):
            conversations = data["conversations"]
            conversations[0]["value"] = conversations[0]["value"].replace("<image>", "<video>")
            video_dataset.append(
                {
                    "videos": [video_path],
                    "conversations": conversations,
                }
            )
        else:
            print(f"Video not found: {video_path}")
            continue
    
    dataset = text_dataset + image_dataset + video_dataset
    random.shuffle(dataset)
    with open("data/datasets/training/train.json", "w") as f:
        json.dump(dataset, f, indent=4)
        
def generate_training_dataset_for_each_modality():
    text_dataset = []
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    dataset = [data for data in dataset]
    random.shuffle(dataset)
    for data in dataset[:10000]:
        conversations = [
            {
                "from": "human",
                "value": data["query"],
            },
            {
                "from": "gpt",
                "value": data["response"],
            },
        ]
        text_dataset.append(
            {
                "conversations": conversations,
            }
        )
    with open("data/datasets/training/text.json", "w") as f:
        json.dump(text_dataset, f, indent=4)
    
    image_dir = "data/datasets/training/VisualWebInstruct/imgs"
    image_dataset = []
    dataset = load_dataset("TIGER-Lab/VisualWebInstruct", "conversation")["train"]
    dataset = [data for data in dataset if data["image"] is not None]
    random.shuffle(dataset)
    for data in dataset:
        if len(image_dataset) >= 10000:
            break
        images = []
        is_complete = True
        for img in data["image"]:
            image_path = os.path.join(image_dir, img)
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                is_complete = False
                break
            images.append(image_path)
        if not is_complete or len(images) > 2:
            continue
        image_dataset.append(
            {
                "images": images,
                "conversations": data["conversations"],
            }
        )
    with open("data/datasets/training/image.json", "w") as f:
        json.dump(image_dataset, f, indent=4)
        
    video_dir = "data/datasets/training/LLaVA-Video-178K"
    video_dataset = []
    dataset = []
    for split in load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1").keys():
        dataset.extend([data for data in load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1")[split]])
    random.shuffle(dataset)
    for data in dataset:
        if len(video_dataset) >= 10000:
            break
        video_path = os.path.join(video_dir, data["data_source"])
        video_path = os.path.join(video_path, data["video"])
        if os.path.exists(video_path):
            conversations = data["conversations"]
            conversations[0]["value"] = conversations[0]["value"].replace("<image>", "<video>")
            video_dataset.append(
                {
                    "videos": [video_path],
                    "conversations": conversations,
                }
            )
        else:
            print(f"Video not found: {video_path}")
            continue
    with open("data/datasets/training/video.json", "w") as f:
        json.dump(video_dataset, f, indent=4)
    
    dataset = text_dataset + image_dataset + video_dataset
    random.shuffle(dataset)
    with open("data/datasets/training/train_multi.json", "w") as f:
        json.dump(dataset, f, indent=4)
        
if __name__ == "__main__":
    # generate_training_dataset()
    # generate_training_dataset_for_each_modality()
    
    with open("data/datasets/training/text.json", "r") as f:
        text_dataset = json.load(f)
    with open("data/datasets/training/image.json", "r") as f:
        image_dataset = json.load(f)
    with open("data/datasets/training/video.json", "r") as f:
        video_dataset = json.load(f)
        
    dataset = text_dataset + image_dataset + video_dataset
    random.shuffle(dataset)
    with open("data/datasets/training/train_multi.json", "w") as f:
        json.dump(dataset, f, indent=4)