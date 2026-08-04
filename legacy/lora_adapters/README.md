# LoRA Adapters for Clinical Reasoning

This directory is the mount point for the dynamically loaded clinical LoRA adapters used by vLLM.
The adapters are built by synthesizing the model weights from the following training outputs:
- `output_from_advanced_training`
- `output_from_training`
- `output_from_training_v2`

For the purpose of this deployment and demo, if the adapter weights are not present here, vLLM will attempt to load the base model `Qwen/Qwen2-7B-Instruct` without adapters, or you can supply the generated LoRA weights into this directory.

## How to Train and Build the LoRA Adapters

To generate the clinical Chain-of-Thought (CoT) dataset from the predictions of the `bki_classifier_advanced_cnn.pt` model and fine-tune Qwen2-7B:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_lora.txt
   ```

2. **Generate the Dataset**:
   This extracts predictions from the CNN model and formats them into a CoT JSONL sequence.
   ```bash
   python generate_lora_dataset.py
   ```
   *Output: `lora_adapters/cot_dataset.jsonl`*

3. **Execute LoRA Fine-Tuning**:
   This runs HuggingFace `SFTTrainer` in `bfloat16` on GPU `cuda:7`, applying a rank-16 LoRA adapter over `q_proj` and `v_proj`.
   ```bash
   python train_lora.py
   ```
   *Output: Adapter weights will be saved to `lora_adapters/clinical-lora/`*

Once training is complete, restarting the `docker-compose` cluster will automatically mount and load these weights into the vLLM engine.
