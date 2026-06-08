import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_base_model(config: dict):
    """
    Load the base Mistral-7B model and tokenizer (without adapter).

    Args:
        config: dict loaded from dpo_config.yaml
    
    Returns:
        model : base mistral model without adapter
    """

    model = AutoModelForCausalLM.from_pretrained(
                                        config["model"]["name"],
                                        dtype = torch.float16,
                                        device_map = "auto")

    return model

def load_dpo_model(config: dict):
    """
    Load the base model and apply the trained DPO adapter from outputs/.

    Args:
        config: dict loaded from dpo_config.yaml
    
    Returns:
        model: base_model wrapped with adapter
    """
    base_model = AutoModelForCausalLM.from_pretrained(
                                            config["model"]["name"],
                                            dtype= torch.float16,
                                            device_map = "auto")

    adapter_model = PeftModel.from_pretrained(base_model, config["adapter"]["dir"])
    
    return adapter_model

def generate_response(model, tokenizer, prompt:str, max_new_tokens:int = 200) -> str:
    """
    Generate a reponse for a given prompt.

    Args:
        model: loaded HF model
        tokenizer: corresponding tokenizer
        prompt: raw string prompt
        max_new_tokens: generation length cap
    
    Returns:
        str: decoded response str (excluding the prompt)
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens = max_new_tokens,eos_token = tokenizer.eos_token_id, pad_token_id = tokenizer.eos_token_id)

    input_length = inputs["input_ids"].shape[1]
    response_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens = True)
    return response

def compare(prompt: str, config: dict) -> dict:
    """
    Run the same prompt throught the base and DPO model, return both outputs.

    Args:
        prompt: str
        config: dict
    
    Returns:
        dict with keys "prompt", "base_response", "dpo_response"
    """

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    base_model = load_base_model(config)
    base_response = generate_response(base_model, tokenizer, prompt)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    adapter_model = load_dpo_model(config)
    dpo_response = generate_response(adapter_model, tokenizer, prompt)
    del adapter_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "prompt" : prompt,
        "base_response" : base_response,
        "dpo_response" : dpo_response
    }

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type = str)

    args = parser.parse_args()

    with open(r"configs/dpo_config.yaml") as f:
        config = yaml.safe_load(f)


    result = compare(args.prompt, config)
    print("-"*100)
    print("===PROMPT===")
    print(result["prompt"])
    print("===BASE===")
    print(result["base_response"])
    print("===DPO===")
    print(result["dpo_response"])
    print("-"*100)