import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

# 加载量化后的模型
model_name = "./bnb_quantized_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用bitsandbytes加载8-bit量化的模型
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 配置8-bit量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',  # 自动分配设备
    torch_dtype=torch.float16,
    quantization_config=quantization_config  # 使用BitsAndBytesConfig进行量化
)

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
with torch.no_grad():
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
