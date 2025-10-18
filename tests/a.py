import kagglehub

# Download latest version
path = kagglehub.model_download("qwen-lm/qwen-3-vl/transformers/235b-a22b-instruct")

print("Path to model files:", path)