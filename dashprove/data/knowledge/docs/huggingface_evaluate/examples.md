Evaluate documentation

🤗 Transformers

# Evaluate

🏡 View all docsAWS Trainium & InferentiaAccelerateArgillaAutoTrainBitsandbytesChat UIDataset
viewerDatasetsDeploying on AWSDiffusersDistilabelEvaluateGoogle CloudGoogle TPUsGradioHubHub Python
LibraryHuggingface.jsInference Endpoints (dedicated)Inference
ProvidersKernelsLeRobotLeaderboardsLightevalMicrosoft AzureOptimumPEFTSafetensorsSentence
TransformersTRLTasksText Embeddings InferenceText Generation
InferenceTokenizersTrackioTransformersTransformers.jssmolagentstimm
Search documentation
mainv0.4.6v0.3.0v0.2.3v0.1.2 EN
Get started
[🤗 Evaluate ][1]
Tutorials
[Installation ][2][A quick tour ][3]
How-to guides
[Choosing the right metric ][4][Adding new evaluations ][5][Using the evaluator ][6][Using the
evaluator with custom pipelines ][7][Creating an EvaluationSuite ][8]
Using 🤗 Evaluate with other ML frameworks
[Transformers ][9][Keras and Tensorflow ][10][scikit-learn ][11]
Conceptual guides
[Types of evaluations ][12][Considerations for model evaluation ][13]
Reference
[Main classes ][14][Loading methods ][15][Saving methods ][16][Hub methods ][17][Evaluator classes
][18][Visualization methods ][19][Logging methods ][20]
[Hugging Face's logo]
Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
[Sign Up][21]

to get started

# 🤗 Transformers

To run the 🤗 Transformers examples make sure you have installed the following libraries:

Copied
pip install datasets transformers torch evaluate nltk rouge_score

## Trainer

The metrics in `evaluate` can be easily integrated with the [Trainer][22]. The `Trainer` accepts a
`compute_metrics` keyword argument that passes a function to compute metrics. One can specify the
evaluation interval with `evaluation_strategy` in the `TrainerArguments`, and based on that, the
model is evaluated accordingly, and the predictions and labels passed to `compute_metrics`.

Copied
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Train
er
import numpy as np
import evaluate

# Prepare and tokenize dataset
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

# Setup evaluation 
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load pretrained model and evaluate model after each epoch
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

## Seq2SeqTrainer

We can use the [Seq2SeqTrainer][23] for sequence-to-sequence tasks such as translation or
summarization. For such generative tasks usually metrics such as ROUGE or BLEU are evaluated.
However, these metrics require that we generate some text with the model rather than a single
forward pass as with e.g. classification. The `Seq2SeqTrainer` allows for the use of the generate
method when setting `predict_with_generate=True` which will generate text for each sample in the
evaluation set. That means we evaluate generated text within the `compute_metric` function. We just
need to decode the predictions and labels first.

Copied
import nltk
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Prepare and tokenize dataset
billsum = load_dataset("billsum", split="ca_test").shuffle(seed=42).select(range(200))
billsum = billsum.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)

# Setup evaluation
nltk.download("punkt_tab", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

# Load pretrained model and evaluate model after each epoch
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    fp16=True,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

You can use any `evaluate` metric with the `Trainer` and `Seq2SeqTrainer` as long as they are
compatible with the task and predictions. In case you don’t want to train a model but just evaluate
an existing model you can replace `trainer.train()` with `trainer.evaluate()` in the above scripts.

[< > Update on GitHub][24]
[←Creating an EvaluationSuite][25] [Keras and Tensorflow→][26]
[🤗 Transformers][27] [Trainer][28] [Seq2SeqTrainer][29]

[1]: /docs/evaluate/index
[2]: /docs/evaluate/installation
[3]: /docs/evaluate/a_quick_tour
[4]: /docs/evaluate/choosing_a_metric
[5]: /docs/evaluate/creating_and_sharing
[6]: /docs/evaluate/base_evaluator
[7]: /docs/evaluate/custom_evaluator
[8]: /docs/evaluate/evaluation_suite
[9]: /docs/evaluate/transformers_integrations
[10]: /docs/evaluate/keras_integrations
[11]: /docs/evaluate/sklearn_integrations
[12]: /docs/evaluate/types_of_evaluations
[13]: /docs/evaluate/considerations
[14]: /docs/evaluate/package_reference/main_classes
[15]: /docs/evaluate/package_reference/loading_methods
[16]: /docs/evaluate/package_reference/saving_methods
[17]: /docs/evaluate/package_reference/hub_methods
[18]: /docs/evaluate/package_reference/evaluator_classes
[19]: /docs/evaluate/package_reference/visualization_methods
[20]: /docs/evaluate/package_reference/logging_methods
[21]: /join
[22]: https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer
[23]: https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Seq2SeqT
rainer
[24]: https://github.com/huggingface/evaluate/blob/main/docs/source/transformers_integrations.mdx
[25]: /docs/evaluate/evaluation_suite
[26]: /docs/evaluate/keras_integrations
[27]: #-transformers
[28]: #trainer
[29]: #seq2seqtrainer
