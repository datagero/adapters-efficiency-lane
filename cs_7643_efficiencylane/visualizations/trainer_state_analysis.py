import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import OrderedDict

def extract_final_metrics(data, adapter=False):
    last_training_entry = None
    last_evaluation_entry = None

    # Iterate backwards over log_history and get the last training and evaluation entries
    for entry in reversed(data['log_history']):
        if 'train_loss' in entry and last_training_entry is None:
            last_training_entry = entry
        if 'eval_loss' in entry and last_evaluation_entry is None:
            last_evaluation_entry = entry

        if adapter and 'loss' in entry and last_training_entry is None:
            # Adapter uses simply loss
            last_training_entry = entry

        # Break the loop if both entries are found
        if last_training_entry is not None and last_evaluation_entry is not None:
            break

    return last_training_entry, last_evaluation_entry

# Define the path to the directory containing your study
training_outputs_path = 'training_output'

training_metrics = []
evaluation_metrics = []

json_dir_dictionary = {}

for study in os.listdir(training_outputs_path):
    study_path = os.path.join(training_outputs_path, study)
    json_dir_dictionary.update({study: {}})
    for trial in os.listdir(study_path):
        json_dir_dictionary[study].update({trial: {}})
        study_dict = json_dir_dictionary[study]

        trial_path = os.path.join(study_path, trial)
        for seed in os.listdir(trial_path):
            seed_path = os.path.join(trial_path, seed)
            
            # If seed_path contains folders that start with checkpoint-
            # then we can assume that the seed_path contains the trainer_state.json file
            if any([folder.startswith('checkpoint-') for folder in os.listdir(seed_path)]):
                # Get latest checkpoint
                checkpoint_folders = [folder for folder in os.listdir(seed_path) if folder.startswith('checkpoint-')]

                # The names of the folders are like checkpoint-7, checkpoint-11.
                # Let's extract the integer and sort by it
                checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split('-')[-1]))
                latest_checkpoint = checkpoint_folders[-1]

                trainer_state_path = os.path.join(seed_path, latest_checkpoint, 'trainer_state.json')
                adapter = True
            else:
                trainer_state_path = os.path.join(seed_path, 'trainer_state.json')
                adapter = False
        
            if os.path.isfile(trainer_state_path):
                study_dict[trial][seed] = {'path': trainer_state_path}

                with open(trainer_state_path, 'r') as json_file:
                    data = json.load(json_file)
                    training_entry, evaluation_entry = extract_final_metrics(data, adapter)

                    if training_entry is None or evaluation_entry is None:
                        # Note, adapters do not have training loss... why???
                        print(f"Skipping {trainer_state_path} due to missing training or evaluation entry")
                        if study == 'roberta-base_citation_intent_pfeiffer_training_base_v01':
                            # For debugging adapter data processing
                            1==1
                        continue

                    study_dict[trial][seed]['training_entry'] = training_entry
                    study_dict[trial][seed]['evaluation_entry'] = evaluation_entry

# Flatten the nested dictionary and construct the dataframe
df_data = []

# Iterate over all trials in the dictionary
for study, study_data in json_dir_dictionary.items():
    for trial, trial_data in study_data.items():
        for seed, seed_data in trial_data.items():
            row = {
                'study': study,
                'trial': trial,
                'seed': seed,
                'path': seed_data['path'],
                **seed_data.get('training_entry', {}),
                **seed_data.get('evaluation_entry', {})
            }
            df_data.append(row)

# Create the dataframe
df = pd.DataFrame(df_data)

# drop path column
df.drop(columns=['path'], inplace=True)
df.head()

abbreviations = OrderedDict([
    ('roberta-base_training_base_v01', 'ROBERTA'),
    ('cs_roberta_base_training_base_v02', 'DAPT'),
    ('dsp_roberta_base_dapt_cs_tapt_citation_intent_1688_training_base_v01', 'DAPT_TAPT'),
    ('dsp_roberta_base_tapt_citation_intent_1688_training_base_v01', 'TAPT'),
    ('mlm_model_training_base_v01', 'MLM-Base'),
    ('roberta-base_citation_intent_pfeiffer_training_base_v01', 'ROBERTA_PFEIFFER'),
    ('roberta-base_citation_intent_seq_bn_training_base_v01', 'ROBERTA_SEQ_BN'),
    ('roberta-base_citation_intent_double_seq_bn_training_base_v01', 'ROBERTA_DOUBLE_SEQ_BN'),
    ('cs_roberta_base_citation_intent_seq_bn_training_base_v01', 'DAPT_SEQ_BN'),
    ('cs_roberta_base_citation_intent_double_seq_bn_training_base_v01', 'DAPT_DOUBLE_SEQ_BN'),
    ('mlm_model_citation_intent_seq_bn_training_base_v01', 'MLM_SEQ_BN'),
    ('mlm_model_citation_intent_double_seq_bn_training_base_v01', 'MLM_DOUBLE_SEQ_BN')
])

comparison_results_path = 'comparative_outputs/dont-stop-pretraining-model-outputs.json'
comparison_results = json.load(open(comparison_results_path))
comparison_results_task_raw = comparison_results['CS']['ACL-ARC']['performance']
comparison_results_task = {k: round(v / 100, 4) for k, v in comparison_results_task_raw.items()}

# Apply the mapping to your 'study' column to create a new 'short_study' column.
df['short_study'] = df['study'].map(abbreviations)

df['trial_order'] = df['trial'].map(lambda x: int(x.split('_')[-1]))
df.sort_values(by=['short_study', 'trial_order'], inplace=True)

# Get the average eval_macro_f1 per trial
df_grouped = df.groupby(['study', 'trial'])['eval_macro_f1'].mean().reset_index()
df = df.merge(df_grouped, how='left', on=['study', 'trial'], suffixes=('', '_av'))
df.rename(columns={'eval_macro_f1_av': 'av_trial_macro_f1'}, inplace=True)

# Now get the max average per study
df_grouped = df.groupby(['short_study'])['av_trial_macro_f1'].mean().reset_index()
df = df.merge(df_grouped, how='left', on=['short_study'], suffixes=('', '_max'))
df.rename(columns={'av_trial_macro_f1_max': 'av_study_macro_f1'}, inplace=True)

# Add benchmarks for adapters
comparison_results_task.update({
    'ROBERTA_PFEIFFER': df[df['short_study'] == 'TAPT']['av_study_macro_f1'].mean(),
    'ROBERTA_SEQ_BN': df[df['short_study'] == 'TAPT']['av_study_macro_f1'].mean(),
    'ROBERTA_DOUBLE_SEQ_BN': df[df['short_study'] == 'TAPT']['av_study_macro_f1'].mean(),
    'DAPT_SEQ_BN': df[df['short_study'] == 'DAPT_TAPT']['av_study_macro_f1'].mean(),
    'DAPT_DOUBLE_SEQ_BN': df[df['short_study'] == 'DAPT_TAPT']['av_study_macro_f1'].mean()})


# Let's assume df is the DataFrame from the previous context
# Generate some visualizations for the DataFrame

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Visualization of Evaluation Loss across different seeds for each trial
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='trial', y='eval_loss', hue='study')
plt.title('Boxplot of Evaluation Loss Across Different Seeds for Each Trial')
plt.xlabel('Trial')
plt.ylabel('Evaluation Loss')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('resources/evaluation_loss_across_seeds.png')
plt.show()

# Visualization of Training Loss across different seeds for each trial
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='trial', y='train_loss', hue='study')
plt.title('Boxplot of Training Loss Across Different Seeds for Each Trial')
plt.xlabel('Trial')
plt.ylabel('Training Loss')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('resources/training_loss_across_seeds.png')
plt.show()


# # Increase the figure size for better clarity
# plt.figure(figsize=(14, 8))
# ax = sns.boxplot(data=df, x='short_study', y='eval_macro_f1', hue='trial', 
#     palette='Set2', order=abbreviations.values())

# # Get unique studies in the same order as they appear on the x-axis
# x_ticks_labels = ax.get_xticklabels()
# studies_in_order = [tick.get_text() for tick in x_ticks_labels]

# # Draw benchmark lines for each study
# for i, study in enumerate(abbreviations.values()):
#     # Assuming benchmarks are aligned with the order of abbreviations.values()
#     benchmark = comparison_results_task.get(study)
#     if benchmark:
#         # Get the x position of the study category and the width of the boxes
#         study_x_position = i
#         width_of_boxes = 0.15  # Adjust width to match your box plot width
        
#         # Draw the horizontal line for the benchmark
#         plt.hlines(benchmark, xmin=study_x_position - width_of_boxes,
#                    xmax=study_x_position + width_of_boxes, color='r', linestyle='--')
        
#         # Add benchmark label
#         plt.text(study_x_position, benchmark, f"{benchmark * 100:.2f}%", color='r', ha='center', va='bottom')

# plt.title('Boxplot of Evaluation Macro F1 Score Across Different Seeds for Each Trial', fontsize=16)
# plt.xlabel('Study', fontsize=14)
# plt.ylabel('Evaluation Macro F1 Score', fontsize=14)
# plt.xticks(rotation=60, ha='right', fontsize=12)
# plt.legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
# plt.tight_layout()
# plt.savefig('resources/evaluation_macro_f1_across_seeds.png', dpi=300)
# plt.show()



# Increase the figure size for better clarity
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=df, x='short_study', y='eval_macro_f1', hue='trial', 
    palette='Set2', order=abbreviations.values())

# Font settings for annotations
fontdict = {'fontsize': 10, 'fontweight': 'bold'}

# Draw the benchmark lines and the average eval_macro_f1 per study/trial
for i, study in enumerate(abbreviations.values()):
    benchmark = comparison_results_task.get(study)
    
    if benchmark:
        # Get the y position for the average eval_macro_f1 per study/trial
        av_f1 = df[df['short_study'] == study]['av_study_macro_f1'].max()

        if study in ['ROBERTA_PFEIFFER', 'ROBERTA_SEQ_BN', 'ROBERTA_DOUBLE_SEQ_BN', 'DAPT_SEQ_BN', 'DAPT_DOUBLE_SEQ_BN']:
            # These are experiments for hyperparameter tuning, so we take the best results per trial (trial is made up of 5 seeds)
            av_f1 = df[df['short_study'] == study]['av_trial_macro_f1'].max()

        # The x position is i (the loop index), adjust as necessary
        x_position = i

        # Draw a line to show the benchmark and average eval_macro_f1
        plt.hlines(benchmark, xmin=x_position - 0.25, xmax=x_position + 0.25, colors='red', linestyles='--', lw=2)
        plt.hlines(av_f1, xmin=x_position - 0.25, xmax=x_position + 0.25, colors='green', linestyles='-', lw=2)

        # Assess which one is bigger
        if benchmark > av_f1:
            benchmark_arrow_style = '<-'
            av_f1_arrow_style = '<-'
        else:
            benchmark_arrow_style = '->'
            av_f1_arrow_style = '->'

        # Annotate the benchmark
        plt.annotate(f'{benchmark:.2f}', (x_position, 0.5 + 0.025), textcoords="offset points", xytext=(0,10),
                     ha='center', va='bottom', color='red', fontweight='bold', fontsize=12,
                     arrowprops=dict(arrowstyle=benchmark_arrow_style, color='red'))

        # Annotate the average eval_macro_f1
        plt.annotate(f'{av_f1:.2f}', (x_position, 0.5 - 0.025), textcoords="offset points", xytext=(0,-15),
                     ha='center', va='top', color='green', fontweight='bold', fontsize=12,
                     arrowprops=dict(arrowstyle=av_f1_arrow_style, color='green'))


benchmark_patch = mpatches.Patch(color='red', label='Benchmark', linestyle='--', linewidth=2)
av_f1_patch = mpatches.Patch(color='green', label='Average eval_macro_f1', linestyle='-', linewidth=2)

# Adding legends to the plot
legend1 = ax.legend(handles=[benchmark_patch, av_f1_patch], loc='upper right', bbox_to_anchor=(1, 1), title="Metrics")
ax.add_artist(legend1)  # Adding the first legend manually so it doesn't disappear when adding the second legend for trials

plt.title('Boxplot of Evaluation Macro F1 Score Across Different Seeds for Each Trial')
plt.xlabel('Study')
plt.ylabel('Evaluation Macro F1 Score')
plt.xticks(rotation=45)
plt.legend(title='Trial', loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('resources/evaluation_macro_f1_across_seeds.png', dpi=300)
plt.show()






# # Scatter plot of Evaluation Samples Per Second against Evaluation Loss for all trials
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='eval_samples_per_second', y='eval_loss', hue='trial', style='seed', s=100)
# plt.title('Scatter Plot of Evaluation Samples Per Second vs. Evaluation Loss')
# plt.xlabel('Evaluation Samples Per Second')
# plt.ylabel('Evaluation Loss')
# plt.legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('resources/evaluation_samples_per_second_vs_loss.png')
# plt.show()

1==1