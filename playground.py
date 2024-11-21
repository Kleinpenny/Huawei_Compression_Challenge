import os

file1_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/bnb_quantized_checkpoint/model-00001-of-00002.safetensors") / (1024 * 1024)
file2_size = os.path.getsize("/home/yxpeng/Projects/Huawei_Compression_Challenge/bnb_quantized_checkpoint/model-00002-of-00002.safetensors") / (1024 * 1024)
print("Model size (MB):", file1_size + file2_size)