import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import os
import torch.nn as nn
import bitsandbytes as bnb

def print_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"Trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params

# Load the tokenizer and model
model_path = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Check file size before quantization
print("Quantization前")
file1_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00001-of-00004.safetensors") / (1024 * 1024)
file2_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00002-of-00004.safetensors") / (1024 * 1024)
file3_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00003-of-00004.safetensors") / (1024 * 1024)
file4_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00004-of-00004.safetensors") / (1024 * 1024)

print("Model size (MB):", round((file1_size + file2_size + file3_size + file4_size), 2))

# Load model
device = 'cuda'
dtype = torch.bfloat16
print("Loading model with 8-bit quantization using bitsandbytes...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,  # Use BitsAndBytesConfig for quantization
    torch_dtype=dtype,
    device_map='auto'   # Automatically map model layers to available devices (GPU)
)

print_model_parameters(model)

# Optionally, you can also quantize specific layers with bitsandbytes
# Example: quantizing the linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module = bnb.nn.Linear8bitLt(module.in_features, module.out_features, has_fp16_weights=False)


print("Model quantized using bitsandbytes.")

print("量化后")

# Save the quantized model
model.save_pretrained("./bnb_quantized_checkpoint")
tokenizer.save_pretrained("./bnb_quantized_checkpoint")

file1_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/bnb_quantized_checkpoint/model-00001-of-00002.safetensors") / (1024 * 1024)
file2_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/bnb_quantized_checkpoint/model-00002-of-00002.safetensors") / (1024 * 1024)
print("Model size (MB):", file1_size + file2_size)
print_model_parameters(model)
