import torch
from transformers import AutoTokenizer, LlamaForCausalLM

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
model_path = "./pruned_checkpoint" #"meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = 'cuda'
dtype = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
print_model_parameters(model)

# Prepare the input text
prompt = 'Complete the paragraph: our solar system is'

#确保 pad_token_id 设置为适当的标记 ID，而非默认值 None 或与 eos_token_id 相同：
tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

#在生成 inputs 时，添加 return_attention_mask=True 参数以显式生成 attention_mask：
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, return_attention_mask=True).to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


max_length = 100
temperature = 0.7  # 控制生成文本的多样性，值越低越确定
top_k = 50         # 在每步选择中，保留概率最高的前K个标记
top_p = 0.9        # 核采样（nucleus sampling），基于累积概率筛选标记

# Generate the output with custom parameters
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    do_sample=True,  # 必须启用采样以应用 temperature 和 top_k/top_p
    pad_token_id=tokenizer.pad_token_id
)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)