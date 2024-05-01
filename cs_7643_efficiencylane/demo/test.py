# Initialize the model and Loader
from transformers import RobertaConfig, TextClassificationPipeline, RobertaForSequenceClassification
from data_loaders.citation_intent_data_loader import CSTasksDataLoader
from adapters import AutoAdapterModel, RobertaAdapterModel
import torch

model_variant = "roberta-base"

# dataset_name = "sciie"
# adapter_path = "adapters/training_output/roberta-base_sciie_double_seq_bn_training_adapter_v01_best/trial_2/seed_9091"

dataset_name = "citation_intent"
adapter_path = "adapters/training_output/roberta-base_citation_intent_seq_bn_training_adapter_v01_best/trial_2/seed_9091"

print("Loading Dataset...")
loader = CSTasksDataLoader(model_name="roberta-base",
                                dataset_name=dataset_name,
                                path=f"data/{dataset_name}/",
                                checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")

dataset = loader.load_dataset(overwrite=False)
num_labels = loader.num_labels
print("num_labels:", num_labels)

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======================================================
# Model & Adapter Config
# ======================================================
# Set up training for the Model and Adapter
config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
)

print("Initialising Model...")
model = RobertaAdapterModel.from_pretrained(model_variant, config=config)
model.to(device)

print("Adding Adapter...")
adapter_name = model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)


# Function to predict labels for a list of texts
def classify_texts(model, tokenizer, texts):
    # Prepare the model input
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids']#.to(model.device)
    attention_mask = encoded_inputs['attention_mask']#.to(model.device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return predictions.cpu().numpy()


import torch

# Get all unique labels in the dataset
unique_labels = torch.unique(torch.tensor(dataset['test']['labels']))
text_by_label = {}

# Extract corresponding texts
for label in unique_labels:
    inx_label = [idx for idx, val in enumerate(dataset['test']['labels']) if val == label.item()]
    text_by_label[label.item()] = [dataset['test']['text'][i] for i in inx_label]

print(text_by_label)

predictions_by_label = {}

# Load the tokenizer from the data loader
tokenizer = loader.tokenizer

# Classify texts and store predictions
for label, texts in text_by_label.items():
    if texts:
        predictions = classify_texts(model, tokenizer, texts)
        predictions_by_label[label] = predictions
    else:
        predictions_by_label[label] = []

# Print predictions for each label
for label, predictions in predictions_by_label.items():
    print(f"Label {label} Predictions:", predictions)

pass