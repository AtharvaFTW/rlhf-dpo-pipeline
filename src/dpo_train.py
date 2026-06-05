from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset as ld
from src.utils import format_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOTrainer, DPOConfig
import yaml
import wandb
import os


load_dotenv()

with open("configs/dpo_config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_dataset(tokenizer):
    """
    Load and prepare Anthropic/hh-rlhf dataset from HuggingFace.
    Returns:
        dataset with "promp", "chosen", "rejected" columns
    """
    dataset = ld("Anthropic/hh-rlhf")
    dataset = dataset["train"].select(range(config["dataset"]["max_samples"]))
    dataset = dataset.map(format_dataset)
    dataset = dataset.filter(lambda x: len(tokenizer(x["prompt"])["input_ids"]) <= config["training"]["max_prompt_length"])

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

def build_dpo_trainer(lora_model, tokenizer, dataset):
    """
    Build the TRL DPOTrainer with model, dataset, and training args.
    Returns:
        configured DPOTrainer.
    """
    dpo_config = DPOConfig(
        output_dir = config["training"]["output_dir"],
        per_device_train_batch_size = config["training"]["per_device_train_batch_size"],
        num_train_epochs = config["training"]["num_train_epochs"],
        gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
        learning_rate = float(config["training"]["learning_rate"]),
        max_length = config["training"]["max_length"]
    )
    trainer = DPOTrainer(
        model = lora_model,
        args = dpo_config,
        processing_class = tokenizer,
        train_dataset = dataset,
        )
    
    return trainer

def train():
    """
    Run the full training loop.
    Calls load_dataset, load_model, build_tpo_trainer in sequence.
    """
    wandb.init(project= "rlhf")
    model, tokenizer = load_model()
    dataset = load_dataset(tokenizer)
    trainer = build_dpo_trainer(model, tokenizer, dataset)
    trainer.train()
    save_path = os.path.join(config["training"]["output_dir"],"final")
    trainer.save_model(save_path)

    artifact = wandb.Artifact(
        name = "rlhf",
        type = "model",
        description = "final"
    )
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    train()
    