* [Docs][1] »
* Models
* [ Edit on GitHub][2]

# Models[¶][3]

A Model defines the neural network’s `forward()` method and encapsulates all of the learnable
parameters in the network. Each model also provides a set of named *architectures* that define the
precise network configuration (e.g., embedding dimension, number of layers, etc.).

Both the model type and architecture are selected via the `--arch` command-line argument. Once
selected, a model may expose additional command-line arguments for further configuration.

Note

All fairseq Models extend [`BaseFairseqModel`][4], which in turn extends [`torch.nn.Module`][5].
Thus any fairseq Model can be used as a stand-alone Module in other PyTorch code.

## Convolutional Neural Networks (CNN)[¶][6]

* *class *`fairseq.models.fconv.``FConvModel`(*encoder*, *decoder*)[[source]][7][¶][8]*
  A fully convolutional model, i.e. a convolutional encoder and a convolutional decoder, as
  described in [“Convolutional Sequence to Sequence Learning” (Gehring et al., 2017)][9].
  
  ──────────┬───────────────────────────────────────────────────────────────────────────────────────
  Parameters│* **encoder** ([*FConvEncoder*][10]) – the encoder                                     
  :         │* **decoder** ([*FConvDecoder*][11]) – the decoder                                     
  ──────────┴───────────────────────────────────────────────────────────────────────────────────────
  
  The Convolutional model provides the following named architectures and command-line arguments:
  
  usage: 
          [--arch {fconv,fconv_iwslt_de_en,fconv_wmt_en_ro,fconv_wmt_en_de,fconv_wmt_en_fr}]
          [--dropout D] [--encoder-embed-dim N] [--encoder-embed-path STR]
          [--encoder-layers EXPR] [--decoder-embed-dim N]
          [--decoder-embed-path STR] [--decoder-layers EXPR]
          [--decoder-out-embed-dim N] [--decoder-attention EXPR]
          [--share-input-output-embed]
  
  ### Named architectures[¶][12]
  
  ──────┬───────────────────────────────────────────────────────────────────────────────────────────
  --arch│Possible choices: fconv, fconv_iwslt_de_en, fconv_wmt_en_ro, fconv_wmt_en_de,              
        │fconv_wmt_en_fr                                                                            
  ──────┴───────────────────────────────────────────────────────────────────────────────────────────
  
  ### Additional command-line arguments[¶][13]
  
  ──────────────────┬───────────────────────────────────────────────────────────────────────────────
  --dropout         │dropout probability                                                            
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --encoder-embed-di│encoder embedding dimension                                                    
  m                 │                                                                               
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --encoder-embed-pa│path to pre-trained encoder embedding                                          
  th                │                                                                               
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --encoder-layers  │encoder layers [(dim, kernel_size), …]                                         
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --decoder-embed-di│decoder embedding dimension                                                    
  m                 │                                                                               
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --decoder-embed-pa│path to pre-trained decoder embedding                                          
  th                │                                                                               
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --decoder-layers  │decoder layers [(dim, kernel_size), …]                                         
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --decoder-out-embe│decoder output embedding dimension                                             
  d-dim             │                                                                               
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --decoder-attentio│decoder attention [True, …]                                                    
  n                 │                                                                               
  ──────────────────┼───────────────────────────────────────────────────────────────────────────────
  --share-input-outp│share input and output embeddings (requires –decoder-out-embed-dim and         
  ut-embed          │–decoder-embed-dim to be equal)                                                
                    │                                                                               
                    │Default: False                                                                 
  ──────────────────┴───────────────────────────────────────────────────────────────────────────────
  
  * *static *`add_args`(*parser*)[[source]][14][¶][15]*
    Add model-specific arguments to the parser.
  
  * *classmethod *`build_model`(*args*, *task*)[[source]][16][¶][17]*
    Build a new model instance.

* *class *`fairseq.models.fconv.``FConvEncoder`(*dictionary*, *embed_dim=512*, *embed_dict=None*,
*max_positions=1024*, *convolutions=((512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*,
*3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*,
*3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*,
*3)*, *(512*, *3))*, *dropout=0.1*)[[source]][18][¶][19]*
  Convolutional encoder consisting of len(convolutions) layers.
  
  ───┬──────────────────────────────────────────────────────────────────────────────────────────────
  Par│* **dictionary** ([*Dictionary*][20]) – encoding dictionary                                   
  ame│* **embed_dim** ([*int*][21]*, **optional*) – embedding dimension                             
  ter│* **embed_dict** ([*str*][22]*, **optional*) – filename from which to load pre-trained        
  s: │  embeddings                                                                                  
     │* **max_positions** ([*int*][23]*, **optional*) – maximum supported input sequence length     
     │* **convolutions** ([*list*][24]*, **optional*) – the convolutional layer structure. Each list
     │  item i corresponds to convolutional layer i. Layers are given as `(out_channels,            
     │  kernel_width, [residual])`. Residual connections are added between layers when `residual=1` 
     │  (which is the default behavior).                                                            
     │* **dropout** ([*float*][25]*, **optional*) – dropout to be applied before each conv layer    
  ───┴──────────────────────────────────────────────────────────────────────────────────────────────
  
  * `forward`(*src_tokens*, *src_lengths*)[[source]][26][¶][27]*
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **src_tokens** (*LongTensor*) – tokens in the source language of shape (batch, src_len)   
    ame│* **src_lengths** (*LongTensor*) – lengths of each source sentence of shape (batch)         
    ter│                                                                                            
    s: │                                                                                            
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│* **encoder_out** (tuple): a tuple with two elements, where the first element is the last   
    urn│  encoder layer’s output and the second element is the same quantity summed with the input  
    s: │  embedding (used for attention). The shape of both tensors is (batch, src_len, embed_dim). 
       │* **encoder_padding_mask** (ByteTensor): the positions of padding elements of shape (batch, 
       │  src_len)                                                                                  
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│[dict][28]                                                                                  
    urn│                                                                                            
    typ│                                                                                            
    e: │                                                                                            
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `max_positions`()[[source]][29][¶][30]*
    Maximum input length supported by the encoder.
  
  * `reorder_encoder_out`(*encoder_out*, *new_order*)[[source]][31][¶][32]*
    Reorder encoder output according to new_order.
    
    ──────────┬─────────────────────────────────────────────────────────────────────────────────────
    Parameters│* **encoder_out** – output from the `forward()` method                               
    :         │* **new_order** (*LongTensor*) – desired order                                       
    ──────────┼─────────────────────────────────────────────────────────────────────────────────────
    Returns:  │encoder_out rearranged according to new_order                                        
    ──────────┴─────────────────────────────────────────────────────────────────────────────────────

* *class *`fairseq.models.fconv.``FConvDecoder`(*dictionary*, *embed_dim=512*, *embed_dict=None*,
*out_embed_dim=256*, *max_positions=1024*, *convolutions=((512*, *3)*, *(512*, *3)*, *(512*, *3)*,
*(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*,
*(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*, *(512*, *3)*,
*(512*, *3)*, *(512*, *3)*, *(512*, *3))*, *attention=True*, *dropout=0.1*, *share_embed=False*,
*positional_embeddings=True*, *adaptive_softmax_cutoff=None*,
*adaptive_softmax_dropout=0.0*)[[source]][33][¶][34]*
  Convolutional decoder
  
  * `forward`(*prev_output_tokens*, *encoder_out=None*, *incremental_state=None*,
  ***unused*)[[source]][35][¶][36]*
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **prev_output_tokens** (*LongTensor*) – shifted output tokens of shape (batch, tgt_len),  
    ame│  for teacher forcing                                                                       
    ter│* **encoder_out** ([*dict*][37]*, **optional*) – output from the encoder, used for          
    s: │  encoder-side attention                                                                    
       │* **incremental_state** ([*dict*][38]*, **optional*) – dictionary used for storing state    
       │  during [Incremental decoding][39]                                                         
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│* the decoder’s output of shape (batch, tgt_len, vocab)                                     
    urn│* a dictionary with any model-specific outputs                                              
    s: │                                                                                            
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│[tuple][40]                                                                                 
    urn│                                                                                            
    typ│                                                                                            
    e: │                                                                                            
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `max_positions`()[[source]][41][¶][42]*
    Maximum output length supported by the decoder.
  
  * `reorder_incremental_state`(*incremental_state*, *new_order*)[[source]][43][¶][44]*
    Reorder incremental state.
    
    This will be called when the order of the input has changed from the previous time step. A
    typical use case is beam search, where the input order changes between time steps based on the
    selection of beams.

## Long Short-Term Memory (LSTM) networks[¶][45]

* *class *`fairseq.models.lstm.``LSTMModel`(*encoder*, *decoder*)[[source]][46][¶][47]*
  * *static *`add_args`(*parser*)[[source]][48][¶][49]*
    Add model-specific arguments to the parser.
  
  * *classmethod *`build_model`(*args*, *task*)[[source]][50][¶][51]*
    Build a new model instance.
  
  * `forward`(*src_tokens*, *src_lengths*, *prev_output_tokens*, *incremental_state:
  Optional[Dict[str*, *Dict[str*, *Optional[torch.Tensor]]]] = None*)[[source]][52][¶][53]*
    Run the forward pass for an encoder-decoder model.
    
    First feed a batch of source tokens through the encoder. Then, feed the encoder output and
    previous decoder outputs (i.e., teacher forcing) to the decoder to produce the next outputs:
    
    encoder_out = self.encoder(src_tokens, src_lengths)
    return self.decoder(prev_output_tokens, encoder_out)
    
    ────┬───────────────────────────────────────────────────────────────────────────────────────────
    Para│* **src_tokens** (*LongTensor*) – tokens in the source language of shape (batch, src_len)  
    mete│* **src_lengths** (*LongTensor*) – source sentence lengths of shape (batch)                
    rs: │* **prev_output_tokens** (*LongTensor*) – previous decoder outputs of shape (batch,        
        │  tgt_len), for teacher forcing                                                            
    ────┼───────────────────────────────────────────────────────────────────────────────────────────
    Retu│* the decoder’s output of shape (batch, tgt_len, vocab)                                    
    rns:│* a dictionary with any model-specific outputs                                             
    ────┼───────────────────────────────────────────────────────────────────────────────────────────
    Retu│[tuple][54]                                                                                
    rn  │                                                                                           
    type│                                                                                           
    :   │                                                                                           
    ────┴───────────────────────────────────────────────────────────────────────────────────────────

* *class *`fairseq.models.lstm.``LSTMEncoder`(*dictionary*, *embed_dim=512*, *hidden_size=512*,
*num_layers=1*, *dropout_in=0.1*, *dropout_out=0.1*, *bidirectional=False*, *left_pad=True*,
*pretrained_embed=None*, *padding_idx=None*, *max_source_positions=100000.0*)[[source]][55][¶][56]*
  LSTM encoder.
  
  * `forward`(*src_tokens: torch.Tensor*, *src_lengths: torch.Tensor*, *enforce_sorted: bool =
  True*)[[source]][57][¶][58]*
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **src_tokens** (*LongTensor*) – tokens in the source language of shape (batch, src_len)   
    ame│* **src_lengths** (*LongTensor*) – lengths of each source sentence of shape (batch)         
    ter│* **enforce_sorted** ([*bool*][59]*, **optional*) – if True, src_tokens is expected to      
    s: │  contain sequences sorted by length in a decreasing order. If False, this condition is not 
       │  required. Default: True.                                                                  
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `max_positions`()[[source]][60][¶][61]*
    Maximum input length supported by the encoder.
  
  * `reorder_encoder_out`(*encoder_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
  torch.Tensor], new_order*)[[source]][62][¶][63]*
    Reorder encoder output according to new_order.
    
    ──────────┬─────────────────────────────────────────────────────────────────────────────────────
    Parameters│* **encoder_out** – output from the `forward()` method                               
    :         │* **new_order** (*LongTensor*) – desired order                                       
    ──────────┼─────────────────────────────────────────────────────────────────────────────────────
    Returns:  │encoder_out rearranged according to new_order                                        
    ──────────┴─────────────────────────────────────────────────────────────────────────────────────

* *class *`fairseq.models.lstm.``LSTMDecoder`(*dictionary*, *embed_dim=512*, *hidden_size=512*,
*out_embed_dim=512*, *num_layers=1*, *dropout_in=0.1*, *dropout_out=0.1*, *attention=True*,
*encoder_output_units=512*, *pretrained_embed=None*, *share_input_output_embed=False*,
*adaptive_softmax_cutoff=None*, *max_target_positions=100000.0*,
*residuals=False*)[[source]][64][¶][65]*
  LSTM decoder.
  
  * `extract_features`(*prev_output_tokens*, *encoder_out: Optional[Tuple[torch.Tensor*,
  *torch.Tensor*, *torch.Tensor*, *torch.Tensor]] = None*, *incremental_state: Optional[Dict[str*,
  *Dict[str*, *Optional[torch.Tensor]]]] = None*)[[source]][66][¶][67]*
    Similar to *forward* but only return features.
  
  * `forward`(*prev_output_tokens*, *encoder_out: Optional[Tuple[torch.Tensor*, *torch.Tensor*,
  *torch.Tensor*, *torch.Tensor]] = None*, *incremental_state: Optional[Dict[str*, *Dict[str*,
  *Optional[torch.Tensor]]]] = None*, *src_lengths: Optional[torch.Tensor] =
  None*)[[source]][68][¶][69]*
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **prev_output_tokens** (*LongTensor*) – shifted output tokens of shape (batch, tgt_len),  
    ame│  for teacher forcing                                                                       
    ter│* **encoder_out** ([*dict*][70]*, **optional*) – output from the encoder, used for          
    s: │  encoder-side attention                                                                    
       │* **incremental_state** ([*dict*][71]*, **optional*) – dictionary used for storing state    
       │  during [Incremental decoding][72]                                                         
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│* the decoder’s output of shape (batch, tgt_len, vocab)                                     
    urn│* a dictionary with any model-specific outputs                                              
    s: │                                                                                            
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│[tuple][73]                                                                                 
    urn│                                                                                            
    typ│                                                                                            
    e: │                                                                                            
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `max_positions`()[[source]][74][¶][75]*
    Maximum output length supported by the decoder.
  
  * `output_layer`(*x*)[[source]][76][¶][77]*
    Project features to the vocabulary size.
  
  * `reorder_incremental_state`(*incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
  new_order: torch.Tensor*)[[source]][78][¶][79]*
    Reorder incremental state.
    
    This will be called when the order of the input has changed from the previous time step. A
    typical use case is beam search, where the input order changes between time steps based on the
    selection of beams.

## Transformer (self-attention) networks[¶][80]

* *class *`fairseq.models.transformer.``TransformerModel`(*args*, *encoder*,
*decoder*)[[source]][81][¶][82]*
  This is the legacy implementation of the transformer model that uses argparse for configuration.
  
  * *classmethod *`add_args`(*parser*)[[source]][83][¶][84]*
    Add model-specific arguments to the parser.
  
  * *classmethod *`build_model`(*args*, *task*)[[source]][85][¶][86]*
    Build a new model instance.

* *class *`fairseq.models.transformer.``TransformerEncoder`(*args*, *dictionary*, *embed_tokens*,
*return_fc=False*)[[source]][87][¶][88]*

* *class *`fairseq.models.transformer.``TransformerDecoder`(*args*, *dictionary*, *embed_tokens*,
*no_encoder_attn=False*, *output_projection=None*)[[source]][89][¶][90]*

## Adding new models[¶][91]

* `fairseq.models.``register_model`(*name*, *dataclass=None*)[[source]][92][¶][93]*
  New model types can be added to fairseq with the [`register_model()`][94] function decorator.
  
  For example:
  
  @register_model('lstm')
  class LSTM(FairseqEncoderDecoderModel):
      (...)
  
  Note
  
  All models must implement the [`BaseFairseqModel`][95] interface. Typically you will extend
  [`FairseqEncoderDecoderModel`][96] for sequence-to-sequence tasks or [`FairseqLanguageModel`][97]
  for language modeling tasks.
  
  ───────────┬───────────────────────────────────────
  Parameters:│**name** ([*str*][98]) – the name of   
             │the model                              
  ───────────┴───────────────────────────────────────

* `fairseq.models.``register_model_architecture`(*model_name*, *arch_name*)[[source]][99][¶][100]*
  New model architectures can be added to fairseq with the [`register_model_architecture()`][101]
  function decorator. After registration, model architectures can be selected with the `--arch`
  command-line argument.
  
  For example:
  
  @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
  def lstm_luong_wmt_en_de(cfg):
      args.encoder_embed_dim = getattr(cfg.model, 'encoder_embed_dim', 1000)
      (...)
  
  The decorated function should take a single argument *cfg*, which is a `omegaconf.DictConfig`. The
  decorated function should modify these arguments in-place to match the desired architecture.
  
  ──────┬───────────────────────────────────────────────────────────────────────────────────────────
  Parame│* **model_name** ([*str*][102]) – the name of the Model (Model must already be registered) 
  ters: │* **arch_name** ([*str*][103]) – the name of the model architecture (`--arch`)             
  ──────┴───────────────────────────────────────────────────────────────────────────────────────────

* *class *`fairseq.models.``BaseFairseqModel`[[source]][104][¶][105]*
  Base class for fairseq models.
  
  * *classmethod *`add_args`(*parser*)[[source]][106][¶][107]*
    Add model-specific arguments to the parser.
  
  * *classmethod *`build_model`(*args*, *task*)[[source]][108][¶][109]*
    Build a new model instance.
  
  * `extract_features`(**args*, ***kwargs*)[[source]][110][¶][111]*
    Similar to *forward* but only return features.
  
  * *classmethod *`from_pretrained`(*model_name_or_path*, *checkpoint_file='model.pt'*,
  *data_name_or_path='.'*, ***kwargs*)[[source]][112][¶][113]*
    Load a `FairseqModel` from a pre-trained model file. Downloads and caches the pre-trained model
    file if needed.
    
    The base implementation returns a `GeneratorHubInterface`, which can be used to generate
    translations or sample from language models. The underlying `FairseqModel` can be accessed via
    the *generator.models* attribute.
    
    Other models may override this to implement custom hub interfaces.
    
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **model_name_or_path** ([*str*][114]) – either the name of a pre-trained model to load or 
    ame│  a path/URL to a pre-trained model state dict                                              
    ter│* **checkpoint_file** ([*str*][115]*, **optional*) – colon-separated list of checkpoint     
    s: │  files in the model archive to ensemble (default: ‘model.pt’)                              
       │* **data_name_or_path** ([*str*][116]*, **optional*) – point args.data to the archive at the
       │  given path/URL. Can start with ‘.’ or ‘./’ to reuse the model archive path.               
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `get_normalized_probs`(*net_output: Tuple[torch.Tensor, Optional[Dict[str,
  List[Optional[torch.Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, torch.Tensor]] =
  None*)[[source]][117][¶][118]*
    Get normalized probabilities (or log probs) from a net’s output.
  
  * `get_normalized_probs_scriptable`(*net_output: Tuple[torch.Tensor, Optional[Dict[str,
  List[Optional[torch.Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, torch.Tensor]] =
  None*)[[source]][119][¶][120]*
    Scriptable helper function for get_normalized_probs in ~BaseFairseqModel
  
  * `get_targets`(*sample*, *net_output*)[[source]][121][¶][122]*
    Get targets from either the sample or the net’s output.
  
  * *classmethod *`hub_models`()[[source]][123][¶][124]*
  
  * `load_state_dict`(*state_dict*, *strict=True*, *model_cfg:
  Optional[omegaconf.dictconfig.DictConfig] = None*, *args: Optional[argparse.Namespace] =
  None*)[[source]][125][¶][126]*
    Copies parameters and buffers from *state_dict* into this module and its descendants.
    
    Overrides the method in `nn.Module`. Compared with that method this additionally “upgrades”
    *state_dicts* from old checkpoints.
  
  * `make_generation_fast_`(***kwargs*)[[source]][127][¶][128]*
    Legacy entry point to optimize model for faster generation. Prefer
    [prepare_for_inference_][129].
  
  * `max_positions`()[[source]][130][¶][131]*
    Maximum length supported by the model.
  
  * `prepare_for_inference_`(*cfg: omegaconf.dictconfig.DictConfig*)[[source]][132][¶][133]*
    Prepare model for inference.
  
  * `prepare_for_onnx_export_`(***kwargs*)[[source]][134][¶][135]*
    Make model exportable via ONNX trace.
  
  * `set_num_updates`(*num_updates*)[[source]][136][¶][137]*
    State from trainer to pass along to model at every update.
  
  * `upgrade_state_dict`(*state_dict*)[[source]][138][¶][139]*
    Upgrade old state dicts to work with newer code.
  
  * `upgrade_state_dict_named`(*state_dict*, *name*)[[source]][140][¶][141]*
    Upgrade old state dicts to work with newer code.
    
    ──────┬─────────────────────────────────────────────────────────────────────────────────────────
    Parame│* **state_dict** ([*dict*][142]) – state dictionary to upgrade, in place                 
    ters: │* **name** ([*str*][143]) – the state dict key corresponding to the current module       
    ──────┴─────────────────────────────────────────────────────────────────────────────────────────

* *class *`fairseq.models.``FairseqEncoderDecoderModel`(*encoder*,
*decoder*)[[source]][144][¶][145]*
  Base class for encoder-decoder models.
  
  ──────────┬───────────────────────────────────────────────────────────────────────────────────────
  Parameters│* **encoder** ([*FairseqEncoder*][146]) – the encoder                                  
  :         │* **decoder** ([*FairseqDecoder*][147]) – the decoder                                  
  ──────────┴───────────────────────────────────────────────────────────────────────────────────────
  
  * `extract_features`(*src_tokens*, *src_lengths*, *prev_output_tokens*,
  ***kwargs*)[[source]][148][¶][149]*
    Similar to *forward* but only return features.
    
    ─────────┬──────────────────────────────────────────────────────────────────────────────────────
    Returns: │* the decoder’s features of shape (batch, tgt_len, embed_dim)                         
             │* a dictionary with any model-specific outputs                                        
    ─────────┼──────────────────────────────────────────────────────────────────────────────────────
    Return   │[tuple][150]                                                                          
    type:    │                                                                                      
    ─────────┴──────────────────────────────────────────────────────────────────────────────────────
  
  * `forward`(*src_tokens*, *src_lengths*, *prev_output_tokens*, ***kwargs*)[[source]][151][¶][152]*
    Run the forward pass for an encoder-decoder model.
    
    First feed a batch of source tokens through the encoder. Then, feed the encoder output and
    previous decoder outputs (i.e., teacher forcing) to the decoder to produce the next outputs:
    
    encoder_out = self.encoder(src_tokens, src_lengths)
    return self.decoder(prev_output_tokens, encoder_out)
    
    ────┬───────────────────────────────────────────────────────────────────────────────────────────
    Para│* **src_tokens** (*LongTensor*) – tokens in the source language of shape (batch, src_len)  
    mete│* **src_lengths** (*LongTensor*) – source sentence lengths of shape (batch)                
    rs: │* **prev_output_tokens** (*LongTensor*) – previous decoder outputs of shape (batch,        
        │  tgt_len), for teacher forcing                                                            
    ────┼───────────────────────────────────────────────────────────────────────────────────────────
    Retu│* the decoder’s output of shape (batch, tgt_len, vocab)                                    
    rns:│* a dictionary with any model-specific outputs                                             
    ────┼───────────────────────────────────────────────────────────────────────────────────────────
    Retu│[tuple][153]                                                                               
    rn  │                                                                                           
    type│                                                                                           
    :   │                                                                                           
    ────┴───────────────────────────────────────────────────────────────────────────────────────────
  
  * `forward_decoder`(*prev_output_tokens*, ***kwargs*)[[source]][154][¶][155]*
  
  * `max_decoder_positions`()[[source]][156][¶][157]*
    Maximum length supported by the decoder.
  
  * `max_positions`()[[source]][158][¶][159]*
    Maximum length supported by the model.
  
  * `output_layer`(*features*, ***kwargs*)[[source]][160][¶][161]*
    Project features to the default output size (typically vocabulary size).

* *class *`fairseq.models.``FairseqEncoderModel`(*encoder*)[[source]][162][¶][163]*
  Base class for encoder-only models.
  
  ───────────┬───────────────────────────────────────────
  Parameters:│**encoder** ([*FairseqEncoder*][164]) – the
             │encoder                                    
  ───────────┴───────────────────────────────────────────
  
  * `forward`(*src_tokens*, *src_lengths*, ***kwargs*)[[source]][165][¶][166]*
    Run the forward pass for a encoder-only model.
    
    Feeds a batch of tokens through the encoder to generate features.
    
    ───────┬────────────────────────────────────────────────────────────────────────────────────────
    Paramet│* **src_tokens** (*LongTensor*) – input tokens of shape (batch, src_len)                
    ers:   │* **src_lengths** (*LongTensor*) – source sentence lengths of shape (batch)             
    ───────┼────────────────────────────────────────────────────────────────────────────────────────
    Returns│the encoder’s output, typically of shape (batch, src_len, features)                     
    :      │                                                                                        
    ───────┴────────────────────────────────────────────────────────────────────────────────────────
  
  * `get_normalized_probs`(*net_output*, *log_probs*, *sample=None*)[[source]][167][¶][168]*
    Get normalized probabilities (or log probs) from a net’s output.
  
  * `max_positions`()[[source]][169][¶][170]*
    Maximum length supported by the model.

* *class *`fairseq.models.``FairseqLanguageModel`(*decoder*)[[source]][171][¶][172]*
  Base class for decoder-only models.
  
  ───────────┬───────────────────────────────────────────
  Parameters:│**decoder** ([*FairseqDecoder*][173]) – the
             │decoder                                    
  ───────────┴───────────────────────────────────────────
  
  * `extract_features`(*src_tokens*, ***kwargs*)[[source]][174][¶][175]*
    Similar to *forward* but only return features.
    
    ─────────┬──────────────────────────────────────────────────────────────────────────────────────
    Returns: │* the decoder’s features of shape (batch, seq_len, embed_dim)                         
             │* a dictionary with any model-specific outputs                                        
    ─────────┼──────────────────────────────────────────────────────────────────────────────────────
    Return   │[tuple][176]                                                                          
    type:    │                                                                                      
    ─────────┴──────────────────────────────────────────────────────────────────────────────────────
  
  * `forward`(*src_tokens*, ***kwargs*)[[source]][177][¶][178]*
    Run the forward pass for a decoder-only model.
    
    Feeds a batch of tokens through the decoder to predict the next tokens.
    
    ──────┬─────────────────────────────────────────────────────────────────────────────────────────
    Parame│* **src_tokens** (*LongTensor*) – tokens on which to condition the decoder, of shape     
    ters: │  (batch, tgt_len)                                                                       
          │* **src_lengths** (*LongTensor*) – source sentence lengths of shape (batch)              
    ──────┼─────────────────────────────────────────────────────────────────────────────────────────
    Return│* the decoder’s output of shape (batch, seq_len, vocab)                                  
    s:    │* a dictionary with any model-specific outputs                                           
    ──────┼─────────────────────────────────────────────────────────────────────────────────────────
    Return│[tuple][179]                                                                             
    type: │                                                                                         
    ──────┴─────────────────────────────────────────────────────────────────────────────────────────
  
  * `forward_decoder`(*prev_output_tokens*, ***kwargs*)[[source]][180][¶][181]*
  
  * `max_decoder_positions`()[[source]][182][¶][183]*
    Maximum length supported by the decoder.
  
  * `max_positions`()[[source]][184][¶][185]*
    Maximum length supported by the model.
  
  * `output_layer`(*features*, ***kwargs*)[[source]][186][¶][187]*
    Project features to the default output size (typically vocabulary size).
  
  * `supported_targets`[¶][188]*

* *class *`fairseq.models.``FairseqMultiModel`(*encoders*, *decoders*)[[source]][189][¶][190]*
  Base class for combining multiple encoder-decoder models.
  
  * *static *`build_shared_embeddings`(*dicts: Dict[str, fairseq.data.dictionary.Dictionary], langs:
  List[str], embed_dim: int, build_embedding: callable, pretrained_embed_path: Optional[str] =
  None*)[[source]][191][¶][192]*
    Helper function to build shared embeddings for a set of languages after checking that all dicts
    corresponding to those languages are equivalent.
    
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **dicts** – Dict of lang_id to its corresponding Dictionary                               
    ame│* **langs** – languages that we want to share embeddings for                                
    ter│* **embed_dim** – embedding dimension                                                       
    s: │* **build_embedding** – callable function to actually build the embedding                   
       │* **pretrained_embed_path** – Optional path to load pretrained embeddings                   
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `decoder`[¶][193]*
  
  * `encoder`[¶][194]*
  
  * `forward`(*src_tokens*, *src_lengths*, *prev_output_tokens*, ***kwargs*)[[source]][195][¶][196]*
    Defines the computation performed at every call.
    
    Should be overridden by all subclasses.
    
    Note
    
    Although the recipe for forward pass needs to be defined within this function, one should call
    the `Module` instance afterwards instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
  
  * `forward_decoder`(*prev_output_tokens*, ***kwargs*)[[source]][197][¶][198]*
  
  * `load_state_dict`(*state_dict*, *strict=True*, *model_cfg=None*, *args:
  Optional[argparse.Namespace] = None*)[[source]][199][¶][200]*
    Copies parameters and buffers from *state_dict* into this module and its descendants.
    
    Overrides the method in `nn.Module`. Compared with that method this additionally “upgrades”
    *state_dicts* from old checkpoints.
  
  * `max_decoder_positions`()[[source]][201][¶][202]*
    Maximum length supported by the decoder.
  
  * `max_positions`()[[source]][203][¶][204]*
    Maximum length supported by the model.

* *class *`fairseq.models.``FairseqEncoder`(*dictionary*)[[source]][205][¶][206]*
  Base class for encoders.
  
  * `forward`(*src_tokens*, *src_lengths=None*, ***kwargs*)[[source]][207][¶][208]*
    ──────┬─────────────────────────────────────────────────────────────────────────────────────────
    Parame│* **src_tokens** (*LongTensor*) – tokens in the source language of shape (batch, src_len)
    ters: │* **src_lengths** (*LongTensor*) – lengths of each source sentence of shape (batch)      
    ──────┴─────────────────────────────────────────────────────────────────────────────────────────
  
  * `forward_torchscript`(*net_input: Dict[str, torch.Tensor]*)[[source]][209][¶][210]*
    A TorchScript-compatible version of forward.
    
    Encoders which use additional arguments may want to override this method for TorchScript
    compatibility.
  
  * `max_positions`()[[source]][211][¶][212]*
    Maximum input length supported by the encoder.
  
  * `reorder_encoder_out`(*encoder_out*, *new_order*)[[source]][213][¶][214]*
    Reorder encoder output according to new_order.
    
    ──────────┬─────────────────────────────────────────────────────────────────────────────────────
    Parameters│* **encoder_out** – output from the `forward()` method                               
    :         │* **new_order** (*LongTensor*) – desired order                                       
    ──────────┼─────────────────────────────────────────────────────────────────────────────────────
    Returns:  │encoder_out rearranged according to new_order                                        
    ──────────┴─────────────────────────────────────────────────────────────────────────────────────
  
  * `set_num_updates`(*num_updates*)[[source]][215][¶][216]*
    State from trainer to pass along to model at every update.
  
  * `upgrade_state_dict_named`(*state_dict*, *name*)[[source]][217][¶][218]*
    Upgrade old state dicts to work with newer code.

* *class *`fairseq.models.``CompositeEncoder`(*encoders*)[[source]][219][¶][220]*
  A wrapper around a dictionary of [`FairseqEncoder`][221] objects.
  
  We run forward on each encoder and return a dictionary of outputs. The first encoder’s dictionary
  is used for initialization.
  
  ───────────┬──────────────────────────────────────────────────────────────────
  Parameters:│**encoders** ([*dict*][222]) – a dictionary of                    
             │[`FairseqEncoder`][223] objects.                                  
  ───────────┴──────────────────────────────────────────────────────────────────
  
  * `forward`(*src_tokens*, *src_lengths*)[[source]][224][¶][225]*
    ──────┬─────────────────────────────────────────────────────────────────────────────────────────
    Parame│* **src_tokens** (*LongTensor*) – tokens in the source language of shape (batch, src_len)
    ters: │* **src_lengths** (*LongTensor*) – lengths of each source sentence of shape (batch)      
    ──────┼─────────────────────────────────────────────────────────────────────────────────────────
    Return│the outputs from each Encoder                                                            
    s:    │                                                                                         
    ──────┼─────────────────────────────────────────────────────────────────────────────────────────
    Return│[dict][226]                                                                              
    type: │                                                                                         
    ──────┴─────────────────────────────────────────────────────────────────────────────────────────
  
  * `max_positions`()[[source]][227][¶][228]*
    Maximum input length supported by the encoder.
  
  * `reorder_encoder_out`(*encoder_out*, *new_order*)[[source]][229][¶][230]*
    Reorder encoder output according to new_order.

* *class *`fairseq.models.``FairseqDecoder`(*dictionary*)[[source]][231][¶][232]*
  Base class for decoders.
  
  * `extract_features`(*prev_output_tokens*, *encoder_out=None*, ***kwargs*)[[source]][233][¶][234]*
    ─────────┬──────────────────────────────────────────────────────────────────────────────────────
    Returns: │* the decoder’s features of shape (batch, tgt_len, embed_dim)                         
             │* a dictionary with any model-specific outputs                                        
    ─────────┼──────────────────────────────────────────────────────────────────────────────────────
    Return   │[tuple][235]                                                                          
    type:    │                                                                                      
    ─────────┴──────────────────────────────────────────────────────────────────────────────────────
  
  * `forward`(*prev_output_tokens*, *encoder_out=None*, ***kwargs*)[[source]][236][¶][237]*
    ─────┬──────────────────────────────────────────────────────────────────────────────────────────
    Param│* **prev_output_tokens** (*LongTensor*) – shifted output tokens of shape (batch, tgt_len),
    eters│  for teacher forcing                                                                     
    :    │* **encoder_out** ([*dict*][238]*, **optional*) – output from the encoder, used for       
         │  encoder-side attention                                                                  
    ─────┼──────────────────────────────────────────────────────────────────────────────────────────
    Retur│* the decoder’s output of shape (batch, tgt_len, vocab)                                   
    ns:  │* a dictionary with any model-specific outputs                                            
    ─────┼──────────────────────────────────────────────────────────────────────────────────────────
    Retur│[tuple][239]                                                                              
    n    │                                                                                          
    type:│                                                                                          
    ─────┴──────────────────────────────────────────────────────────────────────────────────────────
  
  * `get_normalized_probs`(*net_output: Tuple[torch.Tensor, Optional[Dict[str,
  List[Optional[torch.Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, torch.Tensor]] =
  None*)[[source]][240][¶][241]*
    Get normalized probabilities (or log probs) from a net’s output.
  
  * `get_normalized_probs_scriptable`(*net_output: Tuple[torch.Tensor, Optional[Dict[str,
  List[Optional[torch.Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, torch.Tensor]] =
  None*)[[source]][242][¶][243]*
    Get normalized probabilities (or log probs) from a net’s output.
  
  * `max_positions`()[[source]][244][¶][245]*
    Maximum input length supported by the decoder.
  
  * `output_layer`(*features*, ***kwargs*)[[source]][246][¶][247]*
    Project features to the default output size, e.g., vocabulary size.
    
    ───────────┬─────────────────────────────────────────────────────────
    Parameters:│**features** (*Tensor*) – features returned by           
               │*extract_features*.                                      
    ───────────┴─────────────────────────────────────────────────────────
  
  * `upgrade_state_dict_named`(*state_dict*, *name*)[[source]][248][¶][249]*
    Upgrade old state dicts to work with newer code.

## Incremental decoding[¶][250]

* *class *`fairseq.models.``FairseqIncrementalDecoder`(*dictionary*)[[source]][251][¶][252]*
  Base class for incremental decoders.
  
  Incremental decoding is a special mode at inference time where the Model only receives a single
  timestep of input corresponding to the previous output token (for teacher forcing) and must
  produce the next output *incrementally*. Thus the model must cache any long-term state that is
  needed about the sequence, e.g., hidden states, convolutional states, etc.
  
  Compared to the standard [`FairseqDecoder`][253] interface, the incremental decoder interface
  allows [`forward()`][254] functions to take an extra keyword argument (*incremental_state*) that
  can be used to cache state across time-steps.
  
  The [`FairseqIncrementalDecoder`][255] interface also defines the
  [`reorder_incremental_state()`][256] method, which is used during beam search to select and
  reorder the incremental state based on the selection of beams.
  
  To learn more about how incremental decoding works, refer to [this blog][257].
  
  * `extract_features`(*prev_output_tokens*, *encoder_out=None*, *incremental_state=None*,
  ***kwargs*)[[source]][258][¶][259]*
    ─────────┬──────────────────────────────────────────────────────────────────────────────────────
    Returns: │* the decoder’s features of shape (batch, tgt_len, embed_dim)                         
             │* a dictionary with any model-specific outputs                                        
    ─────────┼──────────────────────────────────────────────────────────────────────────────────────
    Return   │[tuple][260]                                                                          
    type:    │                                                                                      
    ─────────┴──────────────────────────────────────────────────────────────────────────────────────
  
  * `forward`(*prev_output_tokens*, *encoder_out=None*, *incremental_state=None*,
  ***kwargs*)[[source]][261][¶][262]*
    ───┬────────────────────────────────────────────────────────────────────────────────────────────
    Par│* **prev_output_tokens** (*LongTensor*) – shifted output tokens of shape (batch, tgt_len),  
    ame│  for teacher forcing                                                                       
    ter│* **encoder_out** ([*dict*][263]*, **optional*) – output from the encoder, used for         
    s: │  encoder-side attention                                                                    
       │* **incremental_state** ([*dict*][264]*, **optional*) – dictionary used for storing state   
       │  during [Incremental decoding][265]                                                        
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│* the decoder’s output of shape (batch, tgt_len, vocab)                                     
    urn│* a dictionary with any model-specific outputs                                              
    s: │                                                                                            
    ───┼────────────────────────────────────────────────────────────────────────────────────────────
    Ret│[tuple][266]                                                                                
    urn│                                                                                            
    typ│                                                                                            
    e: │                                                                                            
    ───┴────────────────────────────────────────────────────────────────────────────────────────────
  
  * `reorder_incremental_state`(*incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
  new_order: torch.Tensor*)[[source]][267][¶][268]*
    Reorder incremental state.
    
    This will be called when the order of the input has changed from the previous time step. A
    typical use case is beam search, where the input order changes between time steps based on the
    selection of beams.
  
  * `reorder_incremental_state_scripting`(*incremental_state: Dict[str, Dict[str,
  Optional[torch.Tensor]]], new_order: torch.Tensor*)[[source]][269][¶][270]*
    Main entry point for reordering the incremental state.
    
    Due to limitations in TorchScript, we call this function in
    `fairseq.sequence_generator.SequenceGenerator` instead of calling
    [`reorder_incremental_state()`][271] directly.
  
  * `set_beam_size`(*beam_size*)[[source]][272][¶][273]*
    Sets the beam size in the decoder and all children.
[Next ][274] [ Previous][275]

© Copyright Facebook AI Research (FAIR) Revision `5ec3a27e`.

Built with [Sphinx][276] using a [theme][277] provided by [Read the Docs][278].

[1]: index.html
[2]: https://github.com/pytorch/fairseq/blob/main/docs/models.rst
[3]: #models
[4]: #fairseq.models.BaseFairseqModel
[5]: https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module
[6]: #module-fairseq.models.fconv
[7]: _modules/fairseq/models/fconv.html#FConvModel
[8]: #fairseq.models.fconv.FConvModel
[9]: https://arxiv.org/abs/1705.03122
[10]: #fairseq.models.fconv.FConvEncoder
[11]: #fairseq.models.fconv.FConvDecoder
[12]: #Named architectures
[13]: #Additional command-line arguments
[14]: _modules/fairseq/models/fconv.html#FConvModel.add_args
[15]: #fairseq.models.fconv.FConvModel.add_args
[16]: _modules/fairseq/models/fconv.html#FConvModel.build_model
[17]: #fairseq.models.fconv.FConvModel.build_model
[18]: _modules/fairseq/models/fconv.html#FConvEncoder
[19]: #fairseq.models.fconv.FConvEncoder
[20]: data.html#fairseq.data.Dictionary
[21]: https://docs.python.org/3/library/functions.html#int
[22]: https://docs.python.org/3/library/stdtypes.html#str
[23]: https://docs.python.org/3/library/functions.html#int
[24]: https://docs.python.org/3/library/stdtypes.html#list
[25]: https://docs.python.org/3/library/functions.html#float
[26]: _modules/fairseq/models/fconv.html#FConvEncoder.forward
[27]: #fairseq.models.fconv.FConvEncoder.forward
[28]: https://docs.python.org/3/library/stdtypes.html#dict
[29]: _modules/fairseq/models/fconv.html#FConvEncoder.max_positions
[30]: #fairseq.models.fconv.FConvEncoder.max_positions
[31]: _modules/fairseq/models/fconv.html#FConvEncoder.reorder_encoder_out
[32]: #fairseq.models.fconv.FConvEncoder.reorder_encoder_out
[33]: _modules/fairseq/models/fconv.html#FConvDecoder
[34]: #fairseq.models.fconv.FConvDecoder
[35]: _modules/fairseq/models/fconv.html#FConvDecoder.forward
[36]: #fairseq.models.fconv.FConvDecoder.forward
[37]: https://docs.python.org/3/library/stdtypes.html#dict
[38]: https://docs.python.org/3/library/stdtypes.html#dict
[39]: #incremental-decoding
[40]: https://docs.python.org/3/library/stdtypes.html#tuple
[41]: _modules/fairseq/models/fconv.html#FConvDecoder.max_positions
[42]: #fairseq.models.fconv.FConvDecoder.max_positions
[43]: _modules/fairseq/models/fconv.html#FConvDecoder.reorder_incremental_state
[44]: #fairseq.models.fconv.FConvDecoder.reorder_incremental_state
[45]: #module-fairseq.models.lstm
[46]: _modules/fairseq/models/lstm.html#LSTMModel
[47]: #fairseq.models.lstm.LSTMModel
[48]: _modules/fairseq/models/lstm.html#LSTMModel.add_args
[49]: #fairseq.models.lstm.LSTMModel.add_args
[50]: _modules/fairseq/models/lstm.html#LSTMModel.build_model
[51]: #fairseq.models.lstm.LSTMModel.build_model
[52]: _modules/fairseq/models/lstm.html#LSTMModel.forward
[53]: #fairseq.models.lstm.LSTMModel.forward
[54]: https://docs.python.org/3/library/stdtypes.html#tuple
[55]: _modules/fairseq/models/lstm.html#LSTMEncoder
[56]: #fairseq.models.lstm.LSTMEncoder
[57]: _modules/fairseq/models/lstm.html#LSTMEncoder.forward
[58]: #fairseq.models.lstm.LSTMEncoder.forward
[59]: https://docs.python.org/3/library/functions.html#bool
[60]: _modules/fairseq/models/lstm.html#LSTMEncoder.max_positions
[61]: #fairseq.models.lstm.LSTMEncoder.max_positions
[62]: _modules/fairseq/models/lstm.html#LSTMEncoder.reorder_encoder_out
[63]: #fairseq.models.lstm.LSTMEncoder.reorder_encoder_out
[64]: _modules/fairseq/models/lstm.html#LSTMDecoder
[65]: #fairseq.models.lstm.LSTMDecoder
[66]: _modules/fairseq/models/lstm.html#LSTMDecoder.extract_features
[67]: #fairseq.models.lstm.LSTMDecoder.extract_features
[68]: _modules/fairseq/models/lstm.html#LSTMDecoder.forward
[69]: #fairseq.models.lstm.LSTMDecoder.forward
[70]: https://docs.python.org/3/library/stdtypes.html#dict
[71]: https://docs.python.org/3/library/stdtypes.html#dict
[72]: #incremental-decoding
[73]: https://docs.python.org/3/library/stdtypes.html#tuple
[74]: _modules/fairseq/models/lstm.html#LSTMDecoder.max_positions
[75]: #fairseq.models.lstm.LSTMDecoder.max_positions
[76]: _modules/fairseq/models/lstm.html#LSTMDecoder.output_layer
[77]: #fairseq.models.lstm.LSTMDecoder.output_layer
[78]: _modules/fairseq/models/lstm.html#LSTMDecoder.reorder_incremental_state
[79]: #fairseq.models.lstm.LSTMDecoder.reorder_incremental_state
[80]: #module-fairseq.models.transformer
[81]: _modules/fairseq/models/transformer/transformer_legacy.html#TransformerModel
[82]: #fairseq.models.transformer.TransformerModel
[83]: _modules/fairseq/models/transformer/transformer_legacy.html#TransformerModel.add_args
[84]: #fairseq.models.transformer.TransformerModel.add_args
[85]: _modules/fairseq/models/transformer/transformer_legacy.html#TransformerModel.build_model
[86]: #fairseq.models.transformer.TransformerModel.build_model
[87]: _modules/fairseq/models/transformer/transformer_encoder.html#TransformerEncoder
[88]: #fairseq.models.transformer.TransformerEncoder
[89]: _modules/fairseq/models/transformer/transformer_decoder.html#TransformerDecoder
[90]: #fairseq.models.transformer.TransformerDecoder
[91]: #adding-new-models
[92]: _modules/fairseq/models.html#register_model
[93]: #fairseq.models.register_model
[94]: #fairseq.models.register_model
[95]: #fairseq.models.BaseFairseqModel
[96]: #fairseq.models.FairseqEncoderDecoderModel
[97]: #fairseq.models.FairseqLanguageModel
[98]: https://docs.python.org/3/library/stdtypes.html#str
[99]: _modules/fairseq/models.html#register_model_architecture
[100]: #fairseq.models.register_model_architecture
[101]: #fairseq.models.register_model_architecture
[102]: https://docs.python.org/3/library/stdtypes.html#str
[103]: https://docs.python.org/3/library/stdtypes.html#str
[104]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel
[105]: #fairseq.models.BaseFairseqModel
[106]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.add_args
[107]: #fairseq.models.BaseFairseqModel.add_args
[108]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.build_model
[109]: #fairseq.models.BaseFairseqModel.build_model
[110]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.extract_features
[111]: #fairseq.models.BaseFairseqModel.extract_features
[112]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.from_pretrained
[113]: #fairseq.models.BaseFairseqModel.from_pretrained
[114]: https://docs.python.org/3/library/stdtypes.html#str
[115]: https://docs.python.org/3/library/stdtypes.html#str
[116]: https://docs.python.org/3/library/stdtypes.html#str
[117]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.get_normalized_probs
[118]: #fairseq.models.BaseFairseqModel.get_normalized_probs
[119]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.get_normalized_probs_scriptable
[120]: #fairseq.models.BaseFairseqModel.get_normalized_probs_scriptable
[121]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.get_targets
[122]: #fairseq.models.BaseFairseqModel.get_targets
[123]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.hub_models
[124]: #fairseq.models.BaseFairseqModel.hub_models
[125]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.load_state_dict
[126]: #fairseq.models.BaseFairseqModel.load_state_dict
[127]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.make_generation_fast_
[128]: #fairseq.models.BaseFairseqModel.make_generation_fast_
[129]: #id3
[130]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.max_positions
[131]: #fairseq.models.BaseFairseqModel.max_positions
[132]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.prepare_for_inference_
[133]: #fairseq.models.BaseFairseqModel.prepare_for_inference_
[134]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.prepare_for_onnx_export_
[135]: #fairseq.models.BaseFairseqModel.prepare_for_onnx_export_
[136]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.set_num_updates
[137]: #fairseq.models.BaseFairseqModel.set_num_updates
[138]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.upgrade_state_dict
[139]: #fairseq.models.BaseFairseqModel.upgrade_state_dict
[140]: _modules/fairseq/models/fairseq_model.html#BaseFairseqModel.upgrade_state_dict_named
[141]: #fairseq.models.BaseFairseqModel.upgrade_state_dict_named
[142]: https://docs.python.org/3/library/stdtypes.html#dict
[143]: https://docs.python.org/3/library/stdtypes.html#str
[144]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel
[145]: #fairseq.models.FairseqEncoderDecoderModel
[146]: #fairseq.models.FairseqEncoder
[147]: #fairseq.models.FairseqDecoder
[148]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel.extract_features
[149]: #fairseq.models.FairseqEncoderDecoderModel.extract_features
[150]: https://docs.python.org/3/library/stdtypes.html#tuple
[151]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel.forward
[152]: #fairseq.models.FairseqEncoderDecoderModel.forward
[153]: https://docs.python.org/3/library/stdtypes.html#tuple
[154]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel.forward_decoder
[155]: #fairseq.models.FairseqEncoderDecoderModel.forward_decoder
[156]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel.max_decoder_positions
[157]: #fairseq.models.FairseqEncoderDecoderModel.max_decoder_positions
[158]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel.max_positions
[159]: #fairseq.models.FairseqEncoderDecoderModel.max_positions
[160]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderDecoderModel.output_layer
[161]: #fairseq.models.FairseqEncoderDecoderModel.output_layer
[162]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderModel
[163]: #fairseq.models.FairseqEncoderModel
[164]: #fairseq.models.FairseqEncoder
[165]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderModel.forward
[166]: #fairseq.models.FairseqEncoderModel.forward
[167]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderModel.get_normalized_probs
[168]: #fairseq.models.FairseqEncoderModel.get_normalized_probs
[169]: _modules/fairseq/models/fairseq_model.html#FairseqEncoderModel.max_positions
[170]: #fairseq.models.FairseqEncoderModel.max_positions
[171]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel
[172]: #fairseq.models.FairseqLanguageModel
[173]: #fairseq.models.FairseqDecoder
[174]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel.extract_features
[175]: #fairseq.models.FairseqLanguageModel.extract_features
[176]: https://docs.python.org/3/library/stdtypes.html#tuple
[177]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel.forward
[178]: #fairseq.models.FairseqLanguageModel.forward
[179]: https://docs.python.org/3/library/stdtypes.html#tuple
[180]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel.forward_decoder
[181]: #fairseq.models.FairseqLanguageModel.forward_decoder
[182]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel.max_decoder_positions
[183]: #fairseq.models.FairseqLanguageModel.max_decoder_positions
[184]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel.max_positions
[185]: #fairseq.models.FairseqLanguageModel.max_positions
[186]: _modules/fairseq/models/fairseq_model.html#FairseqLanguageModel.output_layer
[187]: #fairseq.models.FairseqLanguageModel.output_layer
[188]: #fairseq.models.FairseqLanguageModel.supported_targets
[189]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel
[190]: #fairseq.models.FairseqMultiModel
[191]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel.build_shared_embeddings
[192]: #fairseq.models.FairseqMultiModel.build_shared_embeddings
[193]: #fairseq.models.FairseqMultiModel.decoder
[194]: #fairseq.models.FairseqMultiModel.encoder
[195]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel.forward
[196]: #fairseq.models.FairseqMultiModel.forward
[197]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel.forward_decoder
[198]: #fairseq.models.FairseqMultiModel.forward_decoder
[199]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel.load_state_dict
[200]: #fairseq.models.FairseqMultiModel.load_state_dict
[201]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel.max_decoder_positions
[202]: #fairseq.models.FairseqMultiModel.max_decoder_positions
[203]: _modules/fairseq/models/fairseq_model.html#FairseqMultiModel.max_positions
[204]: #fairseq.models.FairseqMultiModel.max_positions
[205]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder
[206]: #fairseq.models.FairseqEncoder
[207]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder.forward
[208]: #fairseq.models.FairseqEncoder.forward
[209]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder.forward_torchscript
[210]: #fairseq.models.FairseqEncoder.forward_torchscript
[211]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder.max_positions
[212]: #fairseq.models.FairseqEncoder.max_positions
[213]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder.reorder_encoder_out
[214]: #fairseq.models.FairseqEncoder.reorder_encoder_out
[215]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder.set_num_updates
[216]: #fairseq.models.FairseqEncoder.set_num_updates
[217]: _modules/fairseq/models/fairseq_encoder.html#FairseqEncoder.upgrade_state_dict_named
[218]: #fairseq.models.FairseqEncoder.upgrade_state_dict_named
[219]: _modules/fairseq/models/composite_encoder.html#CompositeEncoder
[220]: #fairseq.models.CompositeEncoder
[221]: #fairseq.models.FairseqEncoder
[222]: https://docs.python.org/3/library/stdtypes.html#dict
[223]: #fairseq.models.FairseqEncoder
[224]: _modules/fairseq/models/composite_encoder.html#CompositeEncoder.forward
[225]: #fairseq.models.CompositeEncoder.forward
[226]: https://docs.python.org/3/library/stdtypes.html#dict
[227]: _modules/fairseq/models/composite_encoder.html#CompositeEncoder.max_positions
[228]: #fairseq.models.CompositeEncoder.max_positions
[229]: _modules/fairseq/models/composite_encoder.html#CompositeEncoder.reorder_encoder_out
[230]: #fairseq.models.CompositeEncoder.reorder_encoder_out
[231]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder
[232]: #fairseq.models.FairseqDecoder
[233]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.extract_features
[234]: #fairseq.models.FairseqDecoder.extract_features
[235]: https://docs.python.org/3/library/stdtypes.html#tuple
[236]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.forward
[237]: #fairseq.models.FairseqDecoder.forward
[238]: https://docs.python.org/3/library/stdtypes.html#dict
[239]: https://docs.python.org/3/library/stdtypes.html#tuple
[240]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.get_normalized_probs
[241]: #fairseq.models.FairseqDecoder.get_normalized_probs
[242]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.get_normalized_probs_scriptable
[243]: #fairseq.models.FairseqDecoder.get_normalized_probs_scriptable
[244]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.max_positions
[245]: #fairseq.models.FairseqDecoder.max_positions
[246]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.output_layer
[247]: #fairseq.models.FairseqDecoder.output_layer
[248]: _modules/fairseq/models/fairseq_decoder.html#FairseqDecoder.upgrade_state_dict_named
[249]: #fairseq.models.FairseqDecoder.upgrade_state_dict_named
[250]: #incremental-decoding
[251]: _modules/fairseq/models/fairseq_incremental_decoder.html#FairseqIncrementalDecoder
[252]: #fairseq.models.FairseqIncrementalDecoder
[253]: #fairseq.models.FairseqDecoder
[254]: #fairseq.models.FairseqIncrementalDecoder.forward
[255]: #fairseq.models.FairseqIncrementalDecoder
[256]: #fairseq.models.FairseqIncrementalDecoder.reorder_incremental_state
[257]: http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/
[258]: _modules/fairseq/models/fairseq_incremental_decoder.html#FairseqIncrementalDecoder.extract_fe
atures
[259]: #fairseq.models.FairseqIncrementalDecoder.extract_features
[260]: https://docs.python.org/3/library/stdtypes.html#tuple
[261]: _modules/fairseq/models/fairseq_incremental_decoder.html#FairseqIncrementalDecoder.forward
[262]: #fairseq.models.FairseqIncrementalDecoder.forward
[263]: https://docs.python.org/3/library/stdtypes.html#dict
[264]: https://docs.python.org/3/library/stdtypes.html#dict
[265]: #incremental-decoding
[266]: https://docs.python.org/3/library/stdtypes.html#tuple
[267]: _modules/fairseq/models/fairseq_incremental_decoder.html#FairseqIncrementalDecoder.reorder_in
cremental_state
[268]: #fairseq.models.FairseqIncrementalDecoder.reorder_incremental_state
[269]: _modules/fairseq/models/fairseq_incremental_decoder.html#FairseqIncrementalDecoder.reorder_in
cremental_state_scripting
[270]: #fairseq.models.FairseqIncrementalDecoder.reorder_incremental_state_scripting
[271]: #fairseq.models.FairseqIncrementalDecoder.reorder_incremental_state
[272]: _modules/fairseq/models/fairseq_incremental_decoder.html#FairseqIncrementalDecoder.set_beam_s
ize
[273]: #fairseq.models.FairseqIncrementalDecoder.set_beam_size
[274]: criterions.html
[275]: tasks.html
[276]: http://sphinx-doc.org/
[277]: https://github.com/rtfd/sphinx_rtd_theme
[278]: https://readthedocs.org
