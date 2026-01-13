# Overview
This repository contains artifacts supporting attempts to restrict local LLM output to valid TLA+ syntax.
Two approaches were explored: a [LlamaCpp](https://github.com/ggml-org/llama.cpp) grammar, and the [Guidance](https://github.com/guidance-ai/guidance) library.
The LlamaCpp grammar is the file [`tlaplus-min.gbnf`](tlaplus-min.gbnf) and is used from the command line, while the Guidance approach uses a Jupyter notebook [`tlaplus-guidance.ipynb`](tlaplus-guidance.ipynb).
These prototypes were developed for the [GenAI-Accelerated TLA+ Challenge](https://foundation.tlapl.us/challenge/index.html).

# Dependencies
1. Acquire access to a powerful computer with a large amount of storage and memory; the [DS15_v2](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/dv2-dsv2-series-memory#dsv2-series-11-15) VM model from Microsoft Azure has 20 CPU cores and 140 GB of memory; with a 1 TB block storage disk it can be rented for around $2/hour.
2. Clone this repo to the computer and initialize a python virtual environment by running `python -m venv .` then `source ./bin/activate` to active it.
3. Run `pip install -r requirements.txt` to download & install all required tools & python packages.
4. Run `python download.py` to download the [Llama 3.3 70B model](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF) with BF16 quantization; this requires approximately 134 GB of storage.

# Using the LlamaCpp Grammar
1. Clone the [llama.cpp](https://github.com/ggml-org/llama.cpp) repository and build it
    - Install dev dependencies; on Ubuntu, run `sudo apt update` then `sudo apt install build-essential`
    - From llama.cpp repo root, run `cmake -B build -DLLAMA_CURL=OFF`
    - Run `cmake --build build --config Release -j 20` to build
2. To test whether the TLA+ grammar can parse a real TLA+ file, run `./llama.cpp/build/bin/test-gbnf-validator tlaplus-min.gbnf ModuleName.tla`
    - Note the grammar intentionally does not support all TLA⁺ syntax as language models seem to have trouble with lesser-used TLA⁺ language constructs.
3. To get llama.cpp to generate a TLA+ spec as restricted by the grammar, run `./llama.cpp/build/bin/llama-cli --model Llama-3.3-70B-Instruct/BF16/Llama-3.3-70B-Instruct-BF16-00001-of-00003.gguf --grammar-file tlaplus-min.gbnf --file prompt.txt`
    - The first time this command is run it can take ten minutes or more to load the model into memory; subsequent runs will start quickly due to caching.
    - Customize [`prompt.txt`](prompt.txt) with your own TLA⁺ instructions.

# Using the Guidance Notebook
1. The jupyter server will have been installed by the `pip` instructions up above; follow [this documentation](https://jupyter-server.readthedocs.io/en/latest/operators/public-server.html#jupyter-public-server) to configure the jupyter server to allow remote access.
    - This will require modifying your Azure firewall settings to allow connections on the appropriate port.
    - If TLS encryption is desired, set up an Azure subdomain for your VM then use [`certbot`](https://certbot.eff.org/instructions?ws=other&os=pip) to acquire certificates; use `fullchain.pem` as the certfile and `privkey.pem` as the keyfile. These can be found in `/etc/letsencrypt/live/your.domain.name/` after running `certbot`.
2. Run `jupyter lab` from this repository root then connect via browser using the token provided by the command line output.
    - It can be a good idea to run this in a [tmux](https://github.com/tmux/tmux/wiki) session so it continues running if the SSH connection is lost.
3. Open the `tlaplus-guidance.ipynb` in your browser session and execute the various code boxes; it can take ten minutes or more to load the model into memory the first time.

