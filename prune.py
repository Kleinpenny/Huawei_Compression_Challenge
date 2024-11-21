import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.utils.prune as prune
import os
import torch.nn as nn

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

# 检查文件大小
print("剪枝前")
file1_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00001-of-00004.safetensors") / (1024 * 1024)
file2_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00002-of-00004.safetensors") / (1024 * 1024)
file3_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00003-of-00004.safetensors") / (1024 * 1024)
file4_size = os.path.getsize("/home/yxpeng/DATA/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/model-00004-of-00004.safetensors") / (1024 * 1024)

print("Model size (MB):", round((file1_size + file2_size + file3_size + file4_size), 2))

device = 'cuda'
dtype = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

print_model_parameters(model)


# Apply pruning and sparse conversion
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # Apply pruning
        prune.l1_unstructured(module, name='weight', amount=0.2)  # Prune 20%
        # Remove pruning mask
        prune.remove(module, 'weight')
        # Convert weight to sparse format
        sparse_weight = module.weight.to_sparse()
        # Replace dense weight with sparse weight
        module.weight = nn.Parameter(sparse_weight)

print("剪枝后")

# Save the pruned model
model.save_pretrained("./pruned_checkpoint")
tokenizer.save_pretrained("./pruned_checkpoint")

file1_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/pruned_checkpoint/model-00001-of-00004.safetensors") / (1024 * 1024)
file2_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/pruned_checkpoint/model-00002-of-00004.safetensors") / (1024 * 1024)
file3_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/pruned_checkpoint/model-00003-of-00004.safetensors") / (1024 * 1024)
file4_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/pruned_checkpoint/model-00004-of-00004.safetensors") / (1024 * 1024)
print("Model size (MB):", file1_size + file2_size + file3_size + file4_size)

print_model_parameters(model)
