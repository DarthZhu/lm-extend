import os
import json
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration

ROOT_MODEL_DIRECTORY="data/models"

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
        
def load_weights(base_model_name, pretrained_model_name=None):
    # if no pretrained model, return base model
    if pretrained_model_name is None:
        existing_models = os.listdir(ROOT_MODEL_DIRECTORY)
        if base_model_name not in existing_models:
            raise FileNotFoundError("The base model does not exist in the model directory.")
        base_model = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{base_model_name}"))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{base_model_name}"))
        return base_model, tokenizer
    
    # sanity check
    existing_models = os.listdir(ROOT_MODEL_DIRECTORY)
    if base_model_name not in existing_models or pretrained_model_name not in existing_models:
        raise FileNotFoundError("Either the base model or the multimodal model does not exist in the model directory.")
    
    # load base model from config (no weights)
    base_config = AutoConfig.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{base_model_name}/config.json"))
    pretrained_config = AutoConfig.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{pretrained_model_name}/config.json"))
    # print(base_config)
    # try:
    #     align_config(base_config, pretrained_config)
    # except:
    #     pass
    # base_config.vocab_size = pretrained_config.vocab_size
    # print(base_config)
    # print(pretrained_config)
    # input()
    base_model = AutoModelForCausalLM.from_config(base_config)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{pretrained_model_name}"))
    
    # load weights
    base_model_index = load_index(base_model_name)
    base_model_weight_map = base_model_index.get("weight_map")
    if base_model_weight_map is None:
        raise AssertionError
    load_weight_names = list(base_model_weight_map.keys())
    weight_map = load_tensors(pretrained_model_name, load_weight_names)
    
    # sanity check
    base_model_tensor_names = list(base_model.state_dict().keys())
    load_tensor_names = list(weight_map.keys())
    # for name in base_model_tensor_names:
    #     if name not in load_tensor_names:
    #         raise AssertionError(f"{name} not found in loaded tensors. Will use random tensors!")
    
    # load tensor into model
    base_model_state_dict = base_model.state_dict()
    # print(base_model.state_dict()["lm_head.weight"])
    for name in base_model_tensor_names:
        if name not in load_tensor_names:
            print(f"{name} not found in loaded tensors.")
            continue
        # setattr(base_model_state_dict, name, weight_map[name])
        base_model_state_dict[name] = weight_map[name]
    # print(base_model_state_dict["lm_head.weight"])
    base_model.load_state_dict(base_model_state_dict)
    # print(base_model.state_dict()["lm_head.weight"])
    # input()
    
    return base_model, tokenizer
    
    
if __name__ == "__main__":
    # load_weights("vicuna-7b-v1.5", "liuhaotian/llava-v1.5-7b")
    # model = LlavaForConditionalGeneration.from_pretrained("data/models/LLaVA-NeXT-Video-7B")
    # print(model)
    # tensors = {}
    # with safe_open("data/models/llava-1.5-7b-hf/model.safetensors", framework="pt") as f:
        # print(f)
        # for k in f.keys():
        #     tensors[k] = f.get_tensor(k)
    load_weights("vicuna-7b-v1.5", "llava-1.5-7b-hf")
    # load_weights("Qwen2-7B-Instruct", "Qwen2-VL-7B-Instruct")
    