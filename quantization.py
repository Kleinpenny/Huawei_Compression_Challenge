import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import os
import torch.nn as nn
import torch.quantization as quantization

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
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

print_model_parameters(model)

# Quantization configuration
model.eval()  # Set model to evaluation mode
model.qconfig = quantization.float_qparams_weight_only_qconfig  # Set quantization config for dynamic quantization

# Apply dynamic quantization to the linear layers
quantized_model = quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Only quantize Linear layers
    dtype=torch.qint8  # Quantize to int8
)

print("量化后")

# Save the quantized model
quantized_model.save_pretrained("./quantized_checkpoint")
tokenizer.save_pretrained("./quantized_checkpoint")

file1_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/quantized_checkpoint/model-00001-of-00004.safetensors") / (1024 * 1024)
file2_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/quantized_checkpoint/model-00002-of-00004.safetensors") / (1024 * 1024)
file3_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/quantized_checkpoint/model-00003-of-00004.safetensors") / (1024 * 1024)
file4_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/quantized_checkpoint/model-00004-of-00004.safetensors") / (1024 * 1024)
print("Model size (MB):", file1_size + file2_size + file3_size + file4_size)
print_model_parameters(quantized_model)
