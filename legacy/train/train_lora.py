import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    # Enforce GPU 7
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
    # Swapped from gated google/gemma-2-9b-it to an ungated equivalent model
    model_name = "Qwen/Qwen2-7B-Instruct"
    dataset_path = "lora_adapters/cot_dataset.jsonl"
    output_dir = "lora_adapters/clinical-lora"
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model {model_name} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print("Configuring LoRA (Targeting q_proj, v_proj with r=16)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, # Short run for demo purposes
        save_steps=50,
        bf16=True, # Use bfloat16 as requested by hardware constraints
        optim="adamw_torch",
        report_to="none"
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    print("Starting LoRA fine-tuning...")
    trainer.train()
    
    print(f"Saving final adapter weights to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete! vLLM can now load this adapter dynamically.")

if __name__ == "__main__":
    main()
