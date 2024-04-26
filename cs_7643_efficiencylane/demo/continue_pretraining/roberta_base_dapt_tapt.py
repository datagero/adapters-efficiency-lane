import os
from transformers import Trainer, TrainingArguments
from transformers import RobertaForMaskedLM
import torch

# Our built utilities
from utils import yaml_utils
from data_loaders.citation_intent_data_loader import CSTasksDataLoader

# ======================================================
# Set-up and Load Data
# ======================================================
model_name = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = CSTasksDataLoader(model_name="roberta-base",
                                  dataset_name="citation_intent",
                                  path=f"data/citation_intent/",
                                  checkpoint_path="data/citation_intent/processed_dataset.pt")

dataset = loader.load_dataset(overwrite=False)

# ======================================================
# Model Config & Training
# Note, we build on top of allenai/cs_roberta_base which is already dapt pre-trained
# ======================================================
model = RobertaForMaskedLM.from_pretrained('allenai/cs_roberta_base')
model.to(device)

training_kwargs = yaml_utils.load_training_args_from_yaml('cs_7643_efficiencylane/training_configs/continue_pretraining.yaml')
training_args = TrainingArguments(**training_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=loader.get_data_collator(),
    train_dataset=dataset['train']
    # No need to specify compute_metrics for MLM pretraining
)

# Start training
trainer.train()

output_dir = 'pretrained_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model.save_pretrained(f'./{output_dir}/dapt_tapt_mlm_model')