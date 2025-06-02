import os
import json
import torch
import shutil
from safetensors import safe_open
from transformers import AutoConfig, Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoProcessor

ROOT_MODEL_DIRECTORY="data/models"
MERGED_MODEL_DIRECTORY="data/local_llms"

def load_index(model_name):
    model_directory = os.path.join(ROOT_MODEL_DIRECTORY, model_name)
    # check is bin file or is safetensors file
    if "model.safetensors.index.json" in os.listdir(model_directory):
        with open(os.path.join(model_directory, "model.safetensors.index.json")) as f:
            index = json.load(f)
    elif "pytorch_model.bin.index.json" in os.listdir(model_directory):
        with open(os.path.join(model_directory, "pytorch_model.bin.index.json")) as f:
            index = json.load(f)
    return index

def load_tensors(model_name, load_weight_names=None):
    model_directory = os.path.join(ROOT_MODEL_DIRECTORY, model_name)
    # check is bin file or is safetensors file
    if "model.safetensors.index.json" in os.listdir(model_directory):
        with open(os.path.join(model_directory, "model.safetensors.index.json")) as f:
            index = json.load(f)
        
        # process map file
        tensor_file_map = index.get("weight_map")
        if tensor_file_map is None:
            raise AssertionError("Safetensors index file has no weight map.")
            
        # load necessary weights if specified
        if load_weight_names:
            # find required files
            required_files = []
            for name in load_weight_names:
                # only add "language model" when loading from multimodal model
                if name not in tensor_file_map.keys():
                    name = "language_model." + name
                if name not in tensor_file_map.keys():
                    # raise AssertionError(f"{name} not found in {model_name}.")
                    replaced_name = name.replace("language_model.", "")
                    print(f"{replaced_name} not found in {model_name}.")
                    continue
                required_files.append(tensor_file_map[name])
            required_files = list(set(required_files))
                
            # open required files
            weight_map = {}
            for file in required_files:
                with safe_open(os.path.join(model_directory, file), framework="pt") as f:
                    for k in f.keys():
                        if k in load_weight_names:
                            weight_map[k] = f.get_tensor(k)
                        else:
                            k_ = k.replace("language_model.", "")
                            if k_ in load_weight_names:
                                weight_map[k_] = f.get_tensor(k)
                            
        # load all tensors
        else:
            all_files = []
            for k in tensor_file_map.keys():
                all_files.append(tensor_file_map[k])
            all_files = list(set(all_files))
            
            # open files
            weight_map = {}
            for file in all_files:
                with safe_open(os.path.join(model_directory, file), framework="pt") as f:
                    for k in f.keys():
                        if k in load_weight_names:
                            weight_map[k] = f.get_tensor(k)
        return weight_map
    
    elif "pytorch_model.bin.index.json" in os.listdir(model_directory):
        with open(os.path.join(model_directory, "pytorch_model.bin.index.json")) as f:
            index = json.load(f)
        
        # process map file
        tensor_file_map = index.get("weight_map")
        if tensor_file_map is None:
            raise AssertionError("Safetensors index file has no weight map.")
            
        # load necessary weights if specified
        if load_weight_names:
            # find required files
            required_files = []
            for name in load_weight_names:
                if name not in tensor_file_map.keys():
                    name = "language_model." + name
                if name not in tensor_file_map.keys():
                    # raise AssertionError(f"{name} not found in {model_name}.")
                    print(f"{name} not found in {model_name}.")
                    continue
                required_files.append(tensor_file_map[name])
            required_files = list(set(required_files))
                
            # open required files
            weight_map = {}
            for file in required_files:
                f = torch.load(os.path.join(model_directory, file))
                for k in f.keys():
                    if k in load_weight_names:
                        weight_map[k] = f.get_tensor(k)
                    else:
                        k_ = k.replace("load_weight_names", "")
                        if k_ in load_weight_names:
                            weight_map[k_] = f.get_tensor(k)
        # load all tensors
        else:
            all_files = []
            for k in tensor_file_map.keys():
                all_files.append(tensor_file_map[k])
            all_files = list(set(all_files))
            
            # open files
            weight_map = {}
            for file in all_files:
                f = torch.load(os.path.join(model_directory, file))
                for k in f.keys():
                    if k in load_weight_names:
                        weight_map[k] = f.get_tensor(k)
        return weight_map
    
def align_config(base_model_config, pretrained_model_config):
    base_model_config.bos_token_id = pretrained_model_config.text_config.bos_token_id
    base_model_config.eos_token_id = pretrained_model_config.text_config.eos_token_id
    base_model_config.intermediate_size = pretrained_model_config.text_config.intermediate_size
    base_model_config.vocab_size = pretrained_model_config.text_config.vocab_size
    base_model_config.hidden_size = 4096
    base_model_config.num_attention_heads = 32
    base_model_config.num_key_value_heads = 32
    return base_model_config
        
def save_weights(base_model_name, merged_model_name):
    # load base model from config (no weights)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{base_model_name}"))
    merged_model = AutoModelForCausalLM.from_pretrained(os.path.join(MERGED_MODEL_DIRECTORY, f"{merged_model_name}"))
    processor = AutoProcessor.from_pretrained(os.path.join(MERGED_MODEL_DIRECTORY, f"{merged_model_name}"))
    
    # load tensor into model
    base_model_state_dict = base_model.state_dict()
    merged_model_state_dict = merged_model.state_dict()
    merged_model_tensor_names = list(merged_model_state_dict.keys())
    for name in merged_model_tensor_names:
        base_model_state_dict[name] = merged_model_state_dict[name]
    # print(base_model_state_dict["lm_head.weight"])
    base_model.load_state_dict(base_model_state_dict)
    # print(base_model.state_dict()["lm_head.weight"])
    # input()
    
    output_path = os.path.join(MERGED_MODEL_DIRECTORY, f"{base_model_name}-{merged_model_name}")
    base_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    
    
if __name__ == "__main__":
    # load_weights("vicuna-7b-v1.5", "liuhaotian/llava-v1.5-7b")
    # model = LlavaForConditionalGeneration.from_pretrained("data/models/LLaVA-NeXT-Video-7B")
    # print(model)
    # tensors = {}
    # with safe_open("data/models/llava-1.5-7b-hf/model.safetensors", framework="pt") as f:
        # print(f)
        # for k in f.keys():
        #     tensors[k] = f.get_tensor(k)
    target_model = "Qwen2-merged-weighted-all"
    save_weights("Qwen2-VL-7B-Instruct", target_model)
    
    shutil.copy(os.path.join(ROOT_MODEL_DIRECTORY, "Qwen2-VL-7B-Instruct/preprocessor_config.json"), os.path.join(MERGED_MODEL_DIRECTORY, f"Qwen2-VL-7B-Instruct-{target_model}"))
    