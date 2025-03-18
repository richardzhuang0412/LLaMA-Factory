from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your model and tokenizer
model_path = "saves/OpenThinker-7B-FC-SFT-bf16/checkpoint-410"
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Upload to your HF account
model.push_to_hub("RZ412/OpenThinker-7B-FC-SFT-5-epochs", private=True)
tokenizer.push_to_hub("RZ412/OpenThinker-7B-FC-SFT-5-epochs", private=True)
