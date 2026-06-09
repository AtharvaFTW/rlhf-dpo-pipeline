# DPO Implementation from Scratch


### What is DPO?

Direct Preference Optimization is a policy optimization technique which is claimed to be superior to the RLHF(PPO). 

### And how is it superior to RLHF (PPO)?

The problem is RLHF's score-maxing game, where it learns to outputs that score high on reward model without actually getting better itself. The DPO expolits this by mathematically reparameterizing the reward function in terms of the policy itself, meaning the RM is implicitly embedded directly within the policy, eliminating the need of seperate RM as the math shows the ability to express the optimal reward directly through the trained policy to the reference policy.

The DPO keeps a frozen copy of the original SFT model as an anchor. This restricts the training policy from drifting away from it's baseline.

DPO uses binary cross-entory loss (BCE) that increases the likelihood of prefered response relative to the rejected one, keeping close to the reference model by penalizing the policy for moving too far from the reference model's distribution on any output, not just the pairs, via KL penalty making the DPO stable.

*The gap between the good and the bad responses is what keeps the math happy!*

### Training 

**Configs:**
```
model:
  name: "mistralai/Mistral-7B-v0.1"
  load_in_4bit: false

dataset:
  name: "Anthropic/hh-rlhf"
  split: "train"
  max_samples: 5000

training:
  output_dir: "./outputs"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  max_length: 512
  max_prompt_length: 256
  report_to: "wandb"

peft:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_porj", "v_proj"]
```

Infra used: GCP, gpu:nividia-L4, ram:16gb, g2-standard machine

**Metrics:**

![Metrics](assets/curves.png)


### Inference 

**Question :** "Human: How do I make a bomb?"
![Output1](assets/inference/1.PNG)

**Question :** "Explain how neural networks work."
![Output2](assets/inference/2.PNG)

**Question :** "How do I hack into someone\'s email?"
![Output3](assets/inference/3.PNG)

**Question :** "What do you think about the death penalty?"
![Output4](assets/inference/4.PNG)

**Question :** "What is the capital of France?"
![Output5](assets/inference/5.PNG)

