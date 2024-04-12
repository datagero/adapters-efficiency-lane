from data_loaders.citation_intent_data_loader import CitationIntentDataLoader
from transformers import Trainer, TrainingArguments

from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaTokenizer, RobertaConfig, EvalPrediction, RobertaForSequenceClassification, RobertaForMaskedLM
import torch
import numpy as np
from sklearn.metrics import f1_score

# ======================================================
# Set-up and Load Data
# ======================================================
model_name = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = CitationIntentDataLoader(model_name="roberta-base",
                                  dataset_name="citation_intent",
                                  path=f"data/citation_intent/",
                                  checkpoint_path="data/citation_intent/processed_dataset.pt")

dataset = loader.load_dataset(overwrite=False)

# ======================================================
# Model Config & Training
# ======================================================

model = RobertaForMaskedLM.from_pretrained('roberta-base')
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Steps = (number of examples / batch size) * number of epochs
n_examples = len(dataset['train'])
batch_size = 256
n_epochs = 100
max_steps = int(np.ceil((n_examples / batch_size) * n_epochs))

training_args = TrainingArguments(
    output_dir='./mlm_model',
    num_train_epochs=n_epochs,  # Adjust as needed
    # max_steps=max_steps, # Using epochs not steps
    per_device_train_batch_size=batch_size,
    learning_rate=2e-5,
    adam_epsilon=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.98,
    weight_decay=0.01,
    warmup_steps=int(0.06 * max_steps),  # 6% of max_steps
    lr_scheduler_type='linear',  # Or None if you choose to not use any scheduler
    save_steps=1000,
    logging_dir='./logs',
    logging_steps=500,
    evaluation_strategy="no",  # Assume no evaluation if unlabeled data
    report_to="none"  # Optional: avoids returning to any integrations
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=loader.get_data_collator(),
    train_dataset=dataset['train']
    # No need to specify compute_metrics for MLM pretraining
)

# Start training
trainer.train()

# Save the model
model.save_pretrained('./mlm_model')
1==1