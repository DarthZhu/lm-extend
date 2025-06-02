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
        
def merge_weights(base_model_name, pretrained_model_names=None, method="avg"):
    # if no pretrained model, return base model
    if pretrained_model_names is None:
        existing_models = os.listdir(ROOT_MODEL_DIRECTORY)
        if base_model_name not in existing_models:
            raise FileNotFoundError("The base model does not exist in the model directory.")
        base_model = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{base_model_name}"))
        return base_model
    
    # sanity check
    existing_models = os.listdir(ROOT_MODEL_DIRECTORY)
    if base_model_name in existing_models:
        for model_name in pretrained_model_names:
            if model_name not in existing_models:
                raise FileNotFoundError("Either the extended model does not exist in the model directory.")
    else:
        raise FileNotFoundError("Either the base model does not exist in the model directory.")
    
    # load base model from config (no weights)
    base_config = AutoConfig.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, f"{base_model_name}/config.json"))
    base_model = AutoModelForCausalLM.from_config(base_config)
    
    # load weights
    base_model_index = load_index(base_model_name)
    base_model_weight_map = base_model_index.get("weight_map")
    if base_model_weight_map is None:
        raise AssertionError
    load_weight_names = list(base_model_weight_map.keys())
    weight_maps = [load_tensors(pretrained_model_name, load_weight_names) for pretrained_model_name in pretrained_model_names]
    
    # load tensor into model
    base_model_tensor_names = list(base_model.state_dict().keys())
    base_model_state_dict = base_model.state_dict()
    # print(base_model.state_dict()["lm_head.weight"])
    for name in base_model_tensor_names:
        loaded_tensors = []
        weights = []
        for pretrained_model_name, weight_map in zip(pretrained_model_names, weight_maps):
            load_tensor_names = list(weight_map.keys())
            if name not in load_tensor_names:
                print(f"{name} not found in {pretrained_model_name}.")
                continue
            if pretrained_model_name == base_model_name:
                base_tensor = weight_map[name]
            else:
                weight = torch.mean(torch.abs(weight_map[name] - base_tensor))
                # print(weight)
                weights.append(weight * 1000)
                # weights.append(weight.item())
            loaded_tensors.append(weight_map[name])
        
        # weighted codes
        weights = torch.nn.functional.softmax(torch.tensor(weights)) / 2
        weights = weights.tolist()
        weights.insert(0, 0.5)
        # print(weights)
        # print(loaded_tensors)
        
        # # ablation here
        # weights.insert(0, 1 - weights[0])
        
        # print(weights)
        loaded_tensors = torch.stack(loaded_tensors)   
        weighted_tensors = torch.tensor(weights).view(loaded_tensors.shape[0], *[1] * (loaded_tensors.dim() - 1)) * loaded_tensors
        # print(weighted_tensors)
        merged_tensor = torch.sum(weighted_tensors, dim=0)
        # print(merged_tensor)
        # input()
        base_model_state_dict[name] = merged_tensor
    base_model.load_state_dict(base_model_state_dict)
    
    return base_model

def merge_tokenizers(base_model_name, pretrained_model_names=None):
    base_tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, base_model_name))
    tokenizers = [AutoTokenizer.from_pretrained(os.path.join(ROOT_MODEL_DIRECTORY, name)) for name in pretrained_model_names]
    vocabs = [tok.get_added_vocab() for tok in tokenizers]
    print(vocabs)
    token2id = {}
    new_vocab = []
    for added_vocab in vocabs:
        for token, idx in added_vocab.items():
            if  base_tokenizer.get_vocab().get(token) is None:
                token2id.update({token: idx})
                new_vocab.append(token)
    base_tokenizer.add_tokens(new_vocab)
    for token in new_vocab:
        print(token)
        print(base_tokenizer.convert_tokens_to_ids(token))
        base_tokenizer.vocab[token] = token2id[token]
        print(base_tokenizer.convert_tokens_to_ids(token))
    
    return base_tokenizer   
    
    
if __name__ == "__main__":
    merge_method = "weighted"
    base_model = "Qwen2-7B-Instruct"
    pretrained_model_names = ["Qwen2-7B-Instruct", "Qwen2-VL-7B-Instruct", "LLaVA-Video-7B-Qwen2", "llava-onevision-qwen2-7b-si"] # "Qwen2-VL-7B-Instruct", "LLaVA-Video-7B-Qwen2", 
    model = merge_weights(base_model, pretrained_model_names)
    tokenizer = merge_tokenizers(base_model, pretrained_model_names)
    
    output_path = os.path.join("data/local_llms", f"Qwen2-merged-{merge_method}-all")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)