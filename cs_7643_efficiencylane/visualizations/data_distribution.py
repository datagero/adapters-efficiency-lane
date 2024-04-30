import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Paths to files

datasets_paths = {
    'ACLARC': {
        'dev': 'data/citation_intent/dev.jsonl', 
        'test': 'data/citation_intent/test.jsonl', 
        'train': 'data/citation_intent/train.jsonl'},
    'SCIIE': {
        'dev': 'data/sciie/dev.jsonl',
        'test': 'data/sciie/test.jsonl',
        'train': 'data/sciie/train.jsonl'},
    'CHEMPROT': {
        'dev': 'data/chemprot/dev.jsonl',
        'test': 'data/chemprot/test.jsonl',
        'train': 'data/chemprot/train.jsonl'}
    }



def load_and_analyze(file_path):
    labels = {}
    record_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            record_count += 1
            data = json.loads(line)
            label = data.get("label", None)
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    return record_count, labels

def load_and_analyze(file_path):
    labels = {}
    record_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            record_count += 1
            data = json.loads(line)
            label = data.get("label", None)
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    return record_count, labels

def encode_and_normalize_labels(counts, labels_dict, label_map):
    return {label_map[label]: (count / counts) * 100 for label, count in labels_dict.items()}

datasets_analysis = {}
for key, value in datasets_paths.items():
    datasets_analysis[key] = {}
    for subset, path in value.items():
        count, labels = load_and_analyze(path)
        print(f'{key} {subset} count: {count}')
        print(f'{key} {subset} labels: {labels}')

        label_mapping = {label: i for i, label in enumerate(sorted(labels))}

        datasets_analysis[key][subset] = {
            'count': count,
            'label_distribution': encode_and_normalize_labels(count, labels, label_mapping)
        }

# Make dataset analysis into a dataframe, considering nested keys
datasets_analysis_df = pd.DataFrame([(key, subset, value['count'], value['label_distribution']) for key, values in datasets_analysis.items() for subset, value in values.items()], columns=['Dataset', 'Subset', 'Total Records', 'Label Distribution'])


# Plotting
fig, ax = plt.subplots(figsize=(18, 8))
colors = plt.cm.viridis(np.linspace(0, 1, 20))  # Color map with enough distinct colors
legend_labels = []

for i, row in datasets_analysis_df.iterrows():
    labels = row['Label Distribution']
    bottom = 0
    for label_id, perc in sorted(labels.items()):
        ax.bar(i, perc, bottom=bottom, label=f'{label_id}' if label_id not in legend_labels else "", color=colors[label_id])
        bottom += perc
        if label_id not in legend_labels:
            legend_labels.append(label_id)

    # Add text label for the total records
    ax.text(i, 105, f'Total: {row["Total Records"]}', ha='center', fontsize=14)  # Increase fontsize

# Set the tick positions and labels
ax.set_xticks(np.arange(len(datasets_analysis_df)))
ax.set_xticklabels([])  # Clear existing tick labels

# Show models above the ticks and subsets underneath each model
for i, (model, subset) in enumerate(zip(datasets_analysis_df['Dataset'], datasets_analysis_df['Subset'])):
    if model == 'ACLARC':
        model = 'ACL-ARC'
    if (i-1) % 3 == 0:
        ax.text(i, -4.2, model, ha='center', fontsize=16, weight='bold')  # Increase fontsize
    ax.text(i, -7.1, subset, ha='center', fontsize=12)  # Increase fontsize

# Legend
legend_patches = [Patch(color=colors[label_id], label=str(label_id)) for label_id in legend_labels]
ax.legend(handles=legend_patches, title='Label #', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Increase fontsize

# Labels and titles
ax.set_ylim(0, 120)
ax.set_title('Label Distribution by Percentage Across Tasks and Subsets (Encoded)', fontsize=16)  # Increase fontsize
ax.set_ylabel('Percentage', fontsize=14)  # Increase fontsize
plt.savefig('project_report/resources/label_distribution.png')
plt.show()
pass