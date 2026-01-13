from huggingface_hub import snapshot_download
name = "Llama-3.3-70B-Instruct"
snapshot_download(
  repo_id = f"unsloth/{name}-GGUF",
  local_dir = name,
  allow_patterns=["*BF16*"]
)
