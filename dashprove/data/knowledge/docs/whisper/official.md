# Whisper

[[Blog]][1] [[Paper]][2] [[Model card]][3] [[Colab example]][4]

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse
audio and is also a multitasking model that can perform multilingual speech recognition, speech
translation, and language identification.

## Approach

[[Approach]][5]

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including
multilingual speech recognition, speech translation, spoken language identification, and voice
activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by
the decoder, allowing a single model to replace many stages of a traditional speech-processing
pipeline. The multitask training format uses a set of special tokens that serve as task specifiers
or classification targets.

## Setup

We used Python 3.9.9 and [PyTorch][6] 1.10.1 to train and test our models, but the codebase is
expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also
depends on a few Python packages, most notably [OpenAI's tiktoken][7] for their fast tokenizer
implementation. You can download and install (or update to) the latest release of Whisper with the
following command:

`pip install -U openai-whisper
`

Alternatively, the following command will pull and install the latest commit from this repository,
along with its Python dependencies:

`pip install git+https://github.com/openai/whisper.git 
`

To update the package to the latest version of this repository, please run:

`pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
`

It also requires the command-line tool [`ffmpeg`][8] to be installed on your system, which is
available from most package managers:

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg

You may need [`rust`][9] installed as well, in case [tiktoken][10] does not provide a pre-built
wheel for your platform. If you see installation errors during the `pip install` command above,
please follow the [Getting started page][11] to install Rust development environment. Additionally,
you may need to configure the `PATH` environment variable, e.g. `export
PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`,
you need to install `setuptools_rust`, e.g. by running:

pip install setuptools-rust

## Available models and languages

There are six model sizes, four with English-only versions, offering speed and accuracy tradeoffs.
Below are the names of the available models and their approximate memory requirements and inference
speed relative to the large model. The relative speeds below are measured by transcribing English
speech on a A100, and the real-world speed may vary significantly depending on many factors
including the language, the speaking speed, and the available hardware.

──────┬──────────┬──────────────────┬──────────────────┬─────────────┬──────────────
Size  │Parameters│English-only model│Multilingual model│Required VRAM│Relative speed
──────┼──────────┼──────────────────┼──────────────────┼─────────────┼──────────────
tiny  │39 M      │`tiny.en`         │`tiny`            │~1 GB        │~10x          
──────┼──────────┼──────────────────┼──────────────────┼─────────────┼──────────────
base  │74 M      │`base.en`         │`base`            │~1 GB        │~7x           
──────┼──────────┼──────────────────┼──────────────────┼─────────────┼──────────────
small │244 M     │`small.en`        │`small`           │~2 GB        │~4x           
──────┼──────────┼──────────────────┼──────────────────┼─────────────┼──────────────
medium│769 M     │`medium.en`       │`medium`          │~5 GB        │~2x           
──────┼──────────┼──────────────────┼──────────────────┼─────────────┼──────────────
large │1550 M    │N/A               │`large`           │~10 GB       │1x            
──────┼──────────┼──────────────────┼──────────────────┼─────────────┼──────────────
turbo │809 M     │N/A               │`turbo`           │~6 GB        │~8x           
──────┴──────────┴──────────────────┴──────────────────┴─────────────┴──────────────

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en`
and `base.en` models. We observed that the difference becomes less significant for the `small.en`
and `medium.en` models. Additionally, the `turbo` model is an optimized version of `large-v3` that
offers faster transcription speed with a minimal degradation in accuracy.

Whisper's performance varies widely depending on the language. The figure below shows a performance
breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER
(character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets.
Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix
D.1, D.2, and D.4 of [the paper][12], as well as the BLEU (Bilingual Evaluation Understudy) scores
for translation in Appendix D.3.

[[WER breakdown by language]][13]

## Command-line usage

The following command will transcribe speech in audio files, using the `turbo` model:

whisper audio.flac audio.mp3 audio.wav --model turbo

The default setting (which selects the `turbo` model) works well for transcribing English. However,
**the `turbo` model is not trained for translation tasks**. If you need to **translate non-English
speech into English**, use one of the **multilingual models** (`tiny`, `base`, `small`, `medium`,
`large`) instead of `turbo`.

For example, to transcribe an audio file containing non-English speech, you can specify the
language:

whisper japanese.wav --language Japanese

To **translate** speech into English, use:

whisper japanese.wav --model medium --language Japanese --task translate

> **Note:** The `turbo` model will return the original language even if `--task translate` is
> specified. Use `medium` or `large` for the best translation results.

Run the following to view all available options:

whisper --help

See [tokenizer.py][14] for the list of all available languages.

## Python usage

Transcription can also be performed within Python:

import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding
30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide
lower-level access to the model.

import whisper

model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

## More examples

Please use the [🙌 Show and tell][15] category in Discussions for sharing more example usages of
Whisper and third-party extensions such as web demos, integrations with other tools, ports for
different platforms, etc.

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE][16] for further
details.

[1]: https://openai.com/blog/whisper
[2]: https://arxiv.org/abs/2212.04356
[3]: https://github.com/openai/whisper/blob/main/model-card.md
[4]: https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb
[5]: https://raw.githubusercontent.com/openai/whisper/main/approach.png
[6]: https://pytorch.org/
[7]: https://github.com/openai/tiktoken
[8]: https://ffmpeg.org/
[9]: http://rust-lang.org
[10]: https://github.com/openai/tiktoken
[11]: https://www.rust-lang.org/learn/get-started
[12]: https://arxiv.org/abs/2212.04356
[13]: https://private-user-images.githubusercontent.com/266841/280740425-f4619d66-1058-4005-8f67-a9d
811b77c62.svg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3Lmdpd
Gh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjY0NjQ2NTUsIm5iZiI6MTc2NjQ2NDM1NSwicGF0aCI6I
i8yNjY4NDEvMjgwNzQwNDI1LWY0NjE5ZDY2LTEwNTgtNDAwNS04ZjY3LWE5ZDgxMWI3N2M2Mi5zdmc_WC1BbXotQWxnb3JpdGhtP
UFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjIzJTJGdXMtZWFzd
C0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIyM1QwNDMyMzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16L
VNpZ25hdHVyZT1iZTQwZmQ4MjNlMjY4ZTZkMmFhNzk4MzFiNjlhOTk3ZjM3OGMxNDk3Y2E3NjM2OThkYmQxYWZkNmVhY2Y4NzM0J
lgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.VmdEQ6BwtXxZn-8_ahBBUiveb_dp3M8TaYZrVIjI6Mw
[14]: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
[15]: https://github.com/openai/whisper/discussions/categories/show-and-tell
[16]: https://github.com/openai/whisper/blob/main/LICENSE
