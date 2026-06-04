from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset as ld
from src.utils import format_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model 
import yaml

load_dotenv()

with open("configs/dpo_config.yaml", "r") as f:
    config = yaml.safe_laod(f)

def load_dataset():
    """
    Load and prepare Anthropic/hh-rlhf dataset from HuggingFace.
    Returns:
        dataset with "promp", "chosen", "rejected" columns
    """
    dataset = ld("Anthropic/hh-rlhf")
    dataset = dataset["train"].select(range(config["dataset"]["max_samples"]))
    dataset = dataset.map(format_dataset)

    return dataset


def load_model():
    """
    Load the base model and tokenizer
    Returns:
        model and tokenizer.
    """
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        r = config["peft"]["lora_r"],
        lora_alpha = config["peft"]["lora_alpha"],
        target_modules = config["peft"]["target_modules"],
        lora_dropout= config["peft"]["lora_dropout"]
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
    lora_model = get_peft_model(model, lora_config)
    return lora_model, tokenizer

def build_dpo_trainer():
    """
    Build the TRL DPOTrainer with model, dataset, and training args.
    Returns:
        configured DPOTrainer.
    """
    pass

def train():
    """
    Run the full training loop.
    Calls load_dataset, load_model, build_tpo_trainer in sequence.
    """
    pass

if __name__ == "__main__":
    from src.utils import format_dataset
    res = []
    dataset = load_dataset()
    for i in range(5):
        re = format_dataset(dataset["train"][i])
        res.append(re)
    print(res)
    