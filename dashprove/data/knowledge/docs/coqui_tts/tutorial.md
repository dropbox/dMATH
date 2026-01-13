# Tutorial For Nervous Beginners[#][1]

## Installation[#][2]

User friendly installation. Recommended only for synthesizing voice.

$ pip install TTS

Developer friendly installation.

$ git clone https://github.com/coqui-ai/TTS
$ cd TTS
$ pip install -e .

## Training a `tts` Model[#][3]

A breakdown of a simple script that trains a GlowTTS model on the LJspeech dataset. See the comments
for more details.

### Pure Python Way[#][4]

0. Download your dataset.
   
   In this example, we download and use the LJSpeech dataset. Set the download directory based on
   your preferences.
   
   $ python -c 'from TTS.utils.downloaders import download_ljspeech; download_ljspeech("../recipes/l
   jspeech/");'
1. Define `train.py`.
   
   import os
   
   # Trainer: Where the ✨️ happens.
   # TrainingArgs: Defines the set of arguments of the Trainer.
   from trainer import Trainer, TrainerArgs
   
   # GlowTTSConfig: all model related values for training, validating and testing.
   from TTS.tts.configs.glow_tts_config import GlowTTSConfig
   
   # BaseDatasetConfig: defines name, formatter and path of the dataset.
   from TTS.tts.configs.shared_configs import BaseDatasetConfig
   from TTS.tts.datasets import load_tts_samples
   from TTS.tts.models.glow_tts import GlowTTS
   from TTS.tts.utils.text.tokenizer import TTSTokenizer
   from TTS.utils.audio import AudioProcessor
   
   # we use the same path as this script as our training folder.
   output_path = os.path.dirname(os.path.abspath(__file__))
   
   # DEFINE DATASET CONFIG
   # Set LJSpeech as our target dataset and define its path.
   # You can also use a simple Dict to define the dataset and pass it to your custom formatter.
   dataset_config = BaseDatasetConfig(
       formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "../LJSp
   eech-1.1/")
   )
   
   # INITIALIZE THE TRAINING CONFIGURATION
   # Configure the model. Every config class inherits the BaseTTSConfig.
   config = GlowTTSConfig(
       batch_size=32,
       eval_batch_size=16,
       num_loader_workers=4,
       num_eval_loader_workers=4,
       run_eval=True,
       test_delay_epochs=-1,
       epochs=1000,
       text_cleaner="phoneme_cleaners",
       use_phonemes=True,
       phoneme_language="en-us",
       phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
       print_step=25,
       print_eval=False,
       mixed_precision=True,
       output_path=output_path,
       datasets=[dataset_config],
   )
   
   # INITIALIZE THE AUDIO PROCESSOR
   # Audio processor is used for feature extraction and audio I/O.
   # It mainly serves to the dataloader and the training loggers.
   ap = AudioProcessor.init_from_config(config)
   
   # INITIALIZE THE TOKENIZER
   # Tokenizer is used to convert text to sequences of token IDs.
   # If characters are not defined in the config, default characters are passed to the config
   tokenizer, config = TTSTokenizer.init_from_config(config)
   
   # LOAD DATA SAMPLES
   # Each sample is a list of ```[text, audio_file_path, speaker_name]```
   # You can define your custom sample loader returning the list of samples.
   # Or define your custom formatter and pass it to the `load_tts_samples`.
   # Check `TTS.tts.datasets.load_tts_samples` for more details.
   train_samples, eval_samples = load_tts_samples(
       dataset_config,
       eval_split=True,
       eval_split_max_size=config.eval_split_max_size,
       eval_split_size=config.eval_split_size,
   )
   
   # INITIALIZE THE MODEL
   # Models take a config object and a speaker manager as input
   # Config defines the details of the model like the number of layers, the size of the embedding, e
   tc.
   # Speaker manager is used by multi-speaker models.
   model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
   
   # INITIALIZE THE TRAINER
   # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-prec
   ision training,
   # distributed training, etc.
   trainer = Trainer(
       TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=ev
   al_samples
   )
   
   # AND... 3,2,1... 🚀
   trainer.fit()
2. Run the script.
   
   CUDA_VISIBLE_DEVICES=0 python train.py
   
   * Continue a previous run.
     
     CUDA_VISIBLE_DEVICES=0 python train.py --continue_path path/to/previous/run/folder/
   * Fine-tune a model.
     
     CUDA_VISIBLE_DEVICES=0 python train.py --restore_path path/to/model/checkpoint.pth
   * Run multi-gpu training.
     
     CUDA_VISIBLE_DEVICES=0,1,2 python -m trainer.distribute --script train.py

### CLI Way[#][5]

We still support running training from CLI like in the old days. The same training run can also be
started as follows.

1. Define your `config.json`
   
   {
       "run_name": "my_run",
       "model": "glow_tts",
       "batch_size": 32,
       "eval_batch_size": 16,
       "num_loader_workers": 4,
       "num_eval_loader_workers": 4,
       "run_eval": true,
       "test_delay_epochs": -1,
       "epochs": 1000,
       "text_cleaner": "english_cleaners",
       "use_phonemes": false,
       "phoneme_language": "en-us",
       "phoneme_cache_path": "phoneme_cache",
       "print_step": 25,
       "print_eval": true,
       "mixed_precision": false,
       "output_path": "recipes/ljspeech/glow_tts/",
       "datasets":[{"formatter": "ljspeech", "meta_file_train":"metadata.csv", "path": "recipes/ljsp
   eech/LJSpeech-1.1/"}]
   }
2. Start training.
   
   $ CUDA_VISIBLE_DEVICES="0" python TTS/bin/train_tts.py --config_path config.json

## Training a `vocoder` Model[#][6]

import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = os.path.dirname(os.path.abspath(__file__))

config = HifiganConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_
samples
)
trainer.fit()

❗️ Note that you can also use `train_vocoder.py` as the `tts` models above.

## Synthesizing Speech[#][7]

You can run `tts` and synthesize speech directly on the terminal.

$ tts -h # see the help
$ tts --list_models  # list the available models.

[cli.gif]

You can call `tts-server` to start a local demo server that you can open it on your favorite web
browser and 🗣️.

$ tts-server -h # see the help
$ tts-server --list_models  # list the available models.

[server.gif]

[1]: #tutorial-for-nervous-beginners
[2]: #installation
[3]: #training-a-tts-model
[4]: #pure-python-way
[5]: #cli-way
[6]: #training-a-vocoder-model
[7]: #synthesizing-speech
