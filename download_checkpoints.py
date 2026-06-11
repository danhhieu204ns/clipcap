import kagglehub

# Download latest version
path = kagglehub.dataset_download("hiumaidanh/checkpoints-2")

print("Path to dataset files:", path)