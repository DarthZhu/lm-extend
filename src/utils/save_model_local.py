import os
import argparse

from src.utils.load_weights import load_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--pretrained_model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./data/local_llms")
    
    args = parser.parse_args()
    
    model, tokenizer = load_weights(args.base_model_name, args.pretrained_model_name)
    
    output_path = os.path.join(args.output_dir, args.pretrained_model_name)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    

if __name__  == "__main__":
    main()