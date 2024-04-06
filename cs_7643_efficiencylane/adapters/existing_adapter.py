"""
This is adapted code from https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=huLjPAKHLA1g
to use the newer adapters library and to work with the Rotten Tomatoes dataset.
"""
import torch
from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaTokenizer, RobertaConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Data set-up
dataset = load_dataset("rotten_tomatoes")
# dataset.save_to_disk("tmp/rotten_tomatoes")
dataset.num_rows
dataset['train'][0]


# Encode and process the input data
# The transformers model expects the target class column to be named "labels"
# Transform to pytorch tensors and only output the required columns
dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset['train'][0]


# Set up training for the Model and Adapter
config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=2, # binary classification
)

model = RobertaAdapterModel.from_pretrained("roberta-base", config=config)#.to(device)

# Add a new adapter and a matching classification head
model.add_adapter("rotten_tomatoes", config="seq_bn") #alternatively, onfig="lora")
model.add_classification_head(
    "rotten_tomatoes",
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"}
  )

#  The train_adapter() method does two things:
#     It freezes all weights of the pre-trained model, so only the adapter weights are updated during training.
#     It activates the adapter and the prediction head such that both are used in every forward pass.
# Activate the adapter
model.train_adapter("rotten_tomatoes")


# Note the differences in hyperparameters compared to full fine-tuning. Adapter training usually requires a few more training epochs than full fine-tuning.
training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

# Debug -> Did not work for multi-GPU training, you'll typically use torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel.
# Here's a simple way to handle it for DataParallel:
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)

# Debug, did not work -> New code - wrap collator in a dictionary
# old_collator = trainer.data_collator
# trainer.data_collator = lambda data: dict(old_collator(data))
# End new code

# Start training and evaluate
trainer.train()
trainer.evaluate()


from transformers import TextClassificationPipeline
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=training_args.device.index)
classifier("This is awesome!")

model.save_adapter("./final_adapter", "rotten_tomatoes")

# Check final size of the adapter
# !ls -lh final_adapter

# Share Adapter to adapter hub
# model.push_adapter_to_hub(
#     "my-awesome-adapter",
#     "rotten_tomatoes",
#     adapterhub_tag="sentiment/rotten_tomatoes",
#     datasets_tag="rotten_tomatoes"
# )


1==1

