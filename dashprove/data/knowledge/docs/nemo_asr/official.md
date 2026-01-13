# Automatic Speech Recognition (ASR)[#][1]

Automatic Speech Recognition (ASR), also known as Speech To Text (STT), refers to the problem of
automatically transcribing spoken language. You can use NeMo to transcribe speech using open-sourced
pretrained models in [14+ languages][2], or [train your own][3] ASR models.

## Transcribe speech with 3 lines of code[#][4]

After installing NeMo, you can transcribe an audio file as follows:

import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
transcript = asr_model.transcribe(["path/to/audio_file.wav"])[0].text

### Obtain timestamps[#][5]

Obtaining char(token), word or segment timestamps is also possible with NeMo ASR Models.

Currently, timestamps are available for Parakeet Models with all types of decoders (CTC/RNNT/TDT).
Support for AED models would be added soon.

There are two ways to obtain timestamps: 1. By using the timestamps=True flag in the transcribe
method. 2. For more control over the timestamps, you can update the decoding config to mention type
of timestamps (char, word, segment) and also specify the segment seperators or word seperator for
segment and word level timestamps.

With the timestamps=True flag, you can obtain timestamps for each character in the transcription as
follows:

# import nemo_asr and instantiate asr_model as above
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# specify flag `timestamps=True`
hypotheses = asr_model.transcribe(["path/to/audio_file.wav"], timestamps=True)

# by default, timestamps are enabled for char, word and segment level
word_timestamps = hypotheses[0].timestamp['word'] # word level timestamps for first sample
segment_timestamps = hypotheses[0].timestamp['segment'] # segment level timestamps
char_timestamps = hypotheses[0].timestamp['char'] # char level timestamps

for stamp in segment_timestamps:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")

# segment level timestamps (if model supports Punctuation and Capitalization, segment level timestam
ps are displayed based on punctuation otherwise complete transcription is considered as a single seg
ment)

For more control over the timestamps, you can update the decoding config to mention type of
timestamps (char, word, segment) and also specify the segment seperators or word seperator for
segment and word level timestamps as follows:

# import nemo_asr and instantiate asr_model as above
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# update decoding config to preserve alignments and compute timestamps
# if necessary also update the segment seperators or word seperator for segment and word level times
tamps
from omegaconf import OmegaConf, open_dict
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    decoding_cfg.segment_seperators = [".", "?", "!"]
    decoding_cfg.word_seperator = " "
    asr_model.change_decoding_strategy(decoding_cfg)

# specify flag `return_hypotheses=True``
hypotheses = asr_model.transcribe(["path/to/audio_file.wav"], return_hypotheses=True)

timestamp_dict = hypotheses[0].timestamp # extract timestamps from hypothesis of first (and only) au
dio file
print("Hypothesis contains following timestamp information :", list(timestamp_dict.keys()))

# For a FastConformer model, you can display the word timestamps as follows:
# 80ms is duration of a timestamp at output of the Conformer
time_stride = 8 * asr_model.cfg.preprocessor.window_stride

word_timestamps = timestamp_dict['word']
segment_timestamps = timestamp_dict['segment']

for stamp in word_timestamps:
    start = stamp['start_offset'] * time_stride
    end = stamp['end_offset'] * time_stride
    word = stamp['char'] if 'char' in stamp else stamp['word']

    print(f"Time : {start:0.2f} - {end:0.2f} - {word}")

for stamp in segment_timestamps:
    start = stamp['start_offset'] * time_stride
    end = stamp['end_offset'] * time_stride
    segment = stamp['segment']

    print(f"Time : {start:0.2f} - {end:0.2f} - {segment}")

## Transcribe speech via command line[#][6]

You can also transcribe speech via the command line using the following [script][7], for example:

python <path_to_NeMo>/blob/main/examples/asr/transcribe_speech.py \
    pretrained_name="stt_en_fastconformer_transducer_large" \
    audio_dir=<path_to_audio_dir> # path to dir containing audio files to transcribe

The script will save all transcriptions in a JSONL file where each line corresponds to an audio file
in `<audio_dir>`. This file will correspond to a format that NeMo commonly uses for saving model
predictions, and also for storing input data for training and evaluation. You can learn more about
the format that NeMo uses for these files (which we refer to as “manifest files”) [here][8].

You can also specify the files to be transcribed inside a manifest file, and pass that in using the
argument `dataset_manifest=<path to manifest specifying audio files to transcribe>` instead of
`audio_dir`.

## Improve ASR transcriptions by incorporating a language model (LM)[#][9]

You can often improve transcription accuracy by incorporating a language model to guide the
selection of more probable words in context. Even a simple n-gram language model can yield a
noticeable improvement.

NeMo supports GPU-accelerated language model fusion for all major ASR model types, including CTC,
RNN-T, TDT, and AED. Customization is available during both greedy and beam decoding. After
[training][10] an n-gram LM, you can apply it using the [speech_to_text_eval.py][11] script.

**To configure the evaluation:**

1. Select the pretrained model: Use the pretrained_name option or provide a local path using
   model_path.
2. Set up the N-gram language model: Provide the path to the NGPU-LM model with ngram_lm_model, and
   set LM weight with ngram_lm_alpha.
3. Choose the decoding strategy:
   
   * CTC models: greedy_batch or beam_batch
   * RNN-T models: greedy_batch, malsd_batch, or maes_batch
   * TDT models: greedy_batch or malsd_batch
   * AED models: beam (set beam_size=1 for greedy decoding)
4. Run the evaluation script.

**Example: CTC Greedy Decoding with NGPU-LM**

python examples/asr/speech_to_text_eval.py \
    pretrained_name=nvidia/parakeet-ctc-1.1b \
    amp=false \
    amp_dtype=bfloat16 \
    matmul_precision=high \
    compute_dtype=bfloat16 \
    presort_manifest=true \
    cuda=0 \
    batch_size=32 \
    dataset_manifest=<path to the evaluation JSON manifest file> \
    ctc_decoding.greedy.ngram_lm_model=<path to the .nemo/.ARPA file of the NGPU-LM model> \
    ctc_decoding.greedy.ngram_lm_alpha=0.2 \
    ctc_decoding.greedy.allow_cuda_graphs=True \
    ctc_decoding.strategy="greedy_batch"

**Example: RNN-T Beam Search with NGPU-LM**

python examples/asr/speech_to_text_eval.py \
    pretrained_name=nvidia/parakeet-rnnt-1.1b \
    amp=false \
    amp_dtype=bfloat16 \
    matmul_precision=high \
    compute_dtype=bfloat16 \
    presort_manifest=true \
    cuda=0 \
    batch_size=16 \
    dataset_manifest=<path to the evaluation JSON manifest file> \
    rnnt_decoding.beam.ngram_lm_model=<path to the .nemo/.ARPA file of the NGPU-LM model> \
    rnnt_decoding.beam.ngram_lm_alpha=0.3 \
    rnnt_decoding.beam.beam_size=10 \
    rnnt_decoding.strategy="malsd_batch"

See detailed documentation here: [ASR Language Modeling and Customization][12].

## Use real-time transcription[#][13]

It is possible to use NeMo to transcribe speech in real-time. We provide tutorial notebooks for
[Cache Aware Streaming][14] and [Buffered Streaming][15].

## Try different ASR models[#][16]

NeMo offers a variety of open-sourced pretrained ASR models that vary by model architecture:

* **encoder architecture** (FastConformer, Conformer, Citrinet, etc.),
* **decoder architecture** (Transducer, CTC & hybrid of the two),
* **size** of the model (small, medium, large, etc.).

The pretrained models also vary by:

* **language** (English, Spanish, etc., including some **multilingual** and **code-switching**
  models),
* whether the output text contains **punctuation & capitalization** or not.

The NeMo ASR checkpoints can be found on [HuggingFace][17], or on [NGC][18]. All models released by
the NeMo team can be found on NGC, and some of those are also available on HuggingFace.

All NeMo ASR checkpoints open-sourced by the NeMo team follow the following naming convention:
`stt_{language}_{encoder name}_{decoder name}_{model size}{_optional descriptor}`.

You can load the checkpoints automatically using the `ASRModel.from_pretrained()` class method, for
example:

import nemo.collections.asr as nemo_asr
# model will be fetched from NGC
asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")
# if model name is prepended with "nvidia/", the model will be fetched from huggingface
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_transducer_large")
# you can also load open-sourced NeMo models released by other HF users using:
# asr_model = nemo_asr.models.ASRModel.from_pretrained("<HF username>/<model name>")

See further documentation about [loading checkpoints][19], a full [list][20] of models and their
[benchmark scores][21].

There is also more information about the ASR model architectures available in NeMo [here][22].

## Try out NeMo ASR transcription in your browser[#][23]

You can try out transcription with a NeMo ASR model without leaving your browser, by using the
HuggingFace Space embedded below.

This HuggingFace Space uses [Parakeet TDT 0.6B V2][24], the latest ASR model from NVIDIA NeMo. It
sits at the top of the [HuggingFace OpenASR Leaderboard][25] at time of writing (May 2nd 2025).

## ASR tutorial notebooks[#][26]

Hands-on speech recognition tutorial notebooks can be found under [the ASR tutorials folder][27]. If
you are a beginner to NeMo, consider trying out the [ASR with NeMo][28] tutorial. This and most
other tutorials can be run on Google Colab by specifying the link to the notebooks’ GitHub pages on
Colab.

## ASR model configuration[#][29]

Documentation regarding the configuration files specific to the `nemo_asr` models can be found in
the [Configuration Files][30] section.

## Preparing ASR datasets[#][31]

NeMo includes preprocessing scripts for several common ASR datasets. The [Datasets][32] section
contains instructions on running those scripts. It also includes guidance for creating your own
NeMo-compatible dataset, if you have your own data.

## NeMo ASR Documentation[#][33]

For more information, see additional sections in the ASR docs on the left-hand-side menu or in the
list below:

* [Models][34]
* [Datasets][35]
* [ASR Language Modeling and Customization][36]
* [Checkpoints][37]
* [Scores][38]
* [Scores with Punctuation and Capitalization][39]
* [NeMo ASR Configuration Files][40]
* [NeMo ASR API][41]
* [All Checkpoints][42]
* [Example With MCV][43]

[1]: #automatic-speech-recognition-asr
[2]: results.html#asr-checkpoint-list-by-language
[3]: examples/kinyarwanda_asr.html
[4]: #transcribe-speech-with-3-lines-of-code
[5]: #obtain-timestamps
[6]: #transcribe-speech-via-command-line
[7]: https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py
[8]: datasets.html#section-with-manifest-format-explanation
[9]: #improve-asr-transcriptions-by-incorporating-a-language-model-lm
[10]: asr_customization/ngram_utils.html#train-ngram-lm
[11]: https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_eval.py
[12]: asr_language_modeling_and_customization.html#asr-language-modeling-and-customization
[13]: #use-real-time-transcription
[14]: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_
Streaming.ipynb
[15]: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Buffered_Str
eaming.ipynb
[16]: #try-different-asr-models
[17]: https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia
[18]: https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC
[19]: results.html
[20]: results.html#asr-checkpoint-list-by-language
[21]: scores.html
[22]: models.html
[23]: #try-out-nemo-asr-transcription-in-your-browser
[24]: https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2
[25]: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
[26]: #asr-tutorial-notebooks
[27]: https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr
[28]: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_NeMo.ipynb
[29]: #asr-model-configuration
[30]: configs.html
[31]: #preparing-asr-datasets
[32]: datasets.html
[33]: #nemo-asr-documentation
[34]: models.html
[35]: datasets.html
[36]: asr_language_modeling_and_customization.html
[37]: results.html
[38]: scores.html
[39]: scores.html#scores-with-punctuation-and-capitalization
[40]: configs.html
[41]: api.html
[42]: all_chkpt.html
[43]: examples/kinyarwanda_asr.html
