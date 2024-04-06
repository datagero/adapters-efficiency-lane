"""
This is adapted code from https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=huLjPAKHLA1g
to use the newer adapters library and to work with the Rotten Tomatoes dataset.
"""

from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaTokenizer, RobertaConfig
from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")
dataset.num_rows
dataset['train'][0]

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset['train'][0]



# Training
config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=2, # binary classification
)

model = RobertaAdapterModel.from_pretrained("roberta-base", config=config)


#  The train_adapter() method does two things:

#     It freezes all weights of the pre-trained model, so only the adapter weights are updated during training.
#     It activates the adapter and the prediction head such that both are used in every forward pass.

# Add a new adapter
model.add_adapter("rotten_tomatoes", config="seq_bn") #alternatively, onfig="lora")

# Add a matching classification head
model.add_classification_head(
    "rotten_tomatoes",
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"}
  )

# Activate the adapter
model.train_adapter("rotten_tomatoes")


import numpy as np
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer

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

# Start training
trainer.train()





# config = RobertaConfig.from_pretrained(
#     "roberta-base",
#     num_labels=2,
#     id2label={ 0: "👎", 1: "👍"},
# )
# model = RobertaAdapterModel.from_pretrained(
#     "roberta-base",
#     config=config,
# )



from transformers import AdapterType

# Add a new adapter
model.add_adapter("rotten_tomatoes", AdapterType.text_task)
# Add a matching classification head
model.add_classification_head("rotten_tomatoes", num_labels=2)
# Activate the adapter
model.train_adapter("rotten_tomatoes")


adapter_name = model.load_adapter("adapter-hub/roberta-base-pf/cs_domain", source="hf", config=AdapterConfig(mh_adapter=True))

model.set_active_adapters(adapter_name)
