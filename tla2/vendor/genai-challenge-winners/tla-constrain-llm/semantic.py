import os
import glob
from os.path import join
from huggingface_hub import snapshot_download
from guidance import models, gen, select

model_properties = [
  ('Llama-3.2-3B-Instruct', 'Q8_0', False),
  ('DeepSeek-R1-Distill-Qwen-14B', 'Q4_K_M', False),
  ('Llama-3.3-70B-Instruct', 'BF16', True)
]

def get_gguf_path(model_name, quantization, is_large):
    return (join(model_name, f'{model_name}-{quantization}.gguf') if not is_large
            else join(model_name, quantization, f'{model_name}-{quantization}-00001-of-00003.gguf'))

model_name, quantization, is_large = model_properties[2]

path = model_name
path = snapshot_download(
  repo_id = f'unsloth/{model_name}-GGUF',
  local_dir = model_name,
  allow_patterns=[f'*{quantization}*']
)

gguf_file = get_gguf_path(model_name, quantization, is_large)
print(gguf_file)

model = models.LlamaCpp(gguf_file, n_gpu_layers=-1, n_ctx=4096, echo=False) 

# append text or generations to the model
lm = model + f'Do you want a joke or a poem? A ' + select(['joke', 'poem'])
print(lm + gen(max_tokens=10))
input()

