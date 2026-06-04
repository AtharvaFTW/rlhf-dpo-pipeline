def format_dataset(sample):
    """
    Format a single sample from hh-rlhf into format DPOTrainer expects.
    Each sample needs "prompt", "chosen", "rejected" keys.
    Return a dict with those three keys
    """
    prompt, chosen_response = sample["chosen"].rsplit("\n\nAssistant: ", 1)
    _, rejected_response = sample["rejected"].rsplit("\n\nAssistant: ", 1)

    res = {"prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response}
    
    return res
