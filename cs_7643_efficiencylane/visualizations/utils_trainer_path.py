"""
These are the utility functions that are used to process the training output data and generate visualizations.
Feel free to use them as a reference to build your own utility functions.
"""

import os
import json
import pandas as pd
import optuna
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

trials_dict = {'adapter': 10, 'model': 5}
seeds_dict = {'adapter': 5, 'model': 5}

class TrainerUtilities:
    """
    Parent class
    """

    def __init__(self, trainer_output_path="training_output", storage="sqlite:///db.sqlite3"):
        self.trials_dict = trials_dict
        self.seeds_dict = seeds_dict
        self.trainer_output_path = trainer_output_path
        self.storage = storage

        self.all_studies_dict = self.get_in_scope_studies()
        self.study_paths = os.listdir(trainer_output_path)
        self.optuna_studies = optuna.study.get_all_study_names(storage)

    def container_of_expected_runs(self, dataset="citation_intent", version="v01"):

        bash_commands_dict = {
            f"roberta-base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh roberta-base {dataset} finetuning model_{version}",
            f"cs_roberta_base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/cs_roberta_base {dataset} finetuning model_{version}",
            f"roberta-base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} seq_bn adapter_{dataset} adapter_{version}",
            f"roberta-base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} double_seq_bn adapter_{dataset} adapter_{version}",
            f"cs_roberta_base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/cs_roberta_base {dataset} seq_bn adapter_{dataset} adapter_{version}",
            f"cs_roberta_base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/cs_roberta_base {dataset} double_seq_bn adapter_{dataset} adapter_{version}"
        }

        if dataset == 'citation_intent':
            bash_commands_dict.update({
                f"mlm_model_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh ./mlm_model {dataset} finetuning model_{version}",
                f"dsp_roberta_base_tapt_citation_intent_1688_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_tapt_citation_intent_1688 {dataset} finetuning model_{version}",
                f"dsp_roberta_base_dapt_cs_tapt_citation_intent_1688_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 {dataset} finetuning model_{version}"
            })
        if dataset == 'sciie':
            bash_commands_dict.update({
                f"dsp_roberta_base_tapt_sciie_3219_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_tapt_sciie_3219 {dataset} finetuning model_{version}",
                f"dsp_roberta_base_dapt_cs_tapt_sciie_3219_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219 {dataset} finetuning model_{version}"
            })

        return bash_commands_dict

    def extract_parameters_from_command(self, command_string, model_type):
        # Split the command to isolate the relevant sections
        parts = command_string.split()
        
        if model_type == 'model':
            # Extracting for the standard model training
            model_variant = parts[2]
            dataset_name = parts[3]
            config_name = parts[4]
            adapter_config_name = None
            study_suffix = parts[5]
            version = study_suffix.split('_')[-1]
        elif model_type == 'adapter':
            # Extracting for the adapter training
            model_variant = parts[2]
            dataset_name = parts[3]
            adapter_config_name = parts[4] # ignore to keep consistent
            config_name = parts[5]
            study_suffix = parts[6]
            version = study_suffix.split('_')[-1]

        # Order of parameters in the command string
        return model_variant, dataset_name, adapter_config_name, config_name, version


    def get_in_scope_studies(self):
        study_dict = {}
        for version in ["v01"]:#, "v02", "v03", "v04", "v05", "v06"]:
            "Three versions for citation_intent dataset"
            study_dict.update(self.container_of_expected_runs(dataset="citation_intent", version=version))

        for version in ["v01"]:#v02", "v03", "v04", "v05", "v06"]:
            "Two versions for sciie dataset"
            study_dict.update(self.container_of_expected_runs(dataset="sciie", version=version))

        # Models might stop being traind before adapters
        in_scope = {}
        for study_name, command in study_dict.items():
            model_type = self.get_model_type(study_name)
            version = study_name.split('_')[-1]
            if version in ['v04', 'v05', 'v06']:
                if model_type == 'model':
                    continue
                elif model_type == 'adapter' and 'mlm_model' in command:
                    continue

            in_scope[study_name] = command

        return in_scope

    def get_all_trial_and_seed_paths(self, study_path):
        """
        Get all trial and seed paths for a given study path.
        """
        all_seed_paths = {}
        for trial in os.listdir(study_path):
            all_seed_paths[trial] = []
            trial_path = os.path.join(study_path, trial)
            if not os.path.isdir(trial_path):
                continue
            for seed in os.listdir(trial_path):
                seed_path = os.path.join(trial_path, seed)
                all_seed_paths[trial].append(seed_path)
        return all_seed_paths

    def get_model_type(self, study_name):
        adapter_match = re.search(r'_adapter_v(\d+)$', study_name)
        base_match = re.search(r'_model_v(\d+)$', study_name)

        if adapter_match:
            return 'adapter'
        elif base_match:
            return 'model'
        else:
            return 'unknown'

    def check_study_completion(self):
        all_study_data = {}
        for study_path in self.study_paths:
            all_study_data.update(self.count_completed_seeds(os.path.join(trainer_output_path, study_path)))

        study_completion = self.check_trainer_study_completion(all_study_data)

        # Get the incomplete studies
        complete_studies = []
        for study, status in study_completion.items():
            if status == 'complete':
                complete_studies.append(study)

        optuna_complete_studies = self.get_optuna_completed_studies()

        # check difference
        complete_not_in_optuna = set(complete_studies) - set(optuna_complete_studies)
        optuna_not_in_trainer = set(optuna_complete_studies) - set(complete_studies)

        # add the two to a single set -> To remediate at some point
        remediate_set = complete_not_in_optuna.union(optuna_not_in_trainer)

        return list(set(complete_studies) - remediate_set), remediate_set

    def check_trainer_study_completion(self, study_data):
        """
        Useful function to identify models that failed / are in progress.
        This info can then be used to clean legacy data or to identify studies that need to be re-run.
        The input is an already processed dictionary with the study names as keys and the number of completed seeds as values.
        """
        completion_check = {}
        
        for study_name, trials in study_data.items():
            # Using regex to dynamically check for the ending pattern and extract the version number
            model_type = self.get_model_type(study_name)

            
            if model_type in self.trials_dict:
                # We expect 10 trials for any adapter version and 5 trials for any model version
                expected_trials = self.trials_dict[model_type]
                expected_seeds_per_trial = self.seeds_dict[model_type]
            else:
                completion_check[study_name] = 'incomplete'
                continue  # Skip further checks as the study name pattern does not match

            # if study_name == "roberta-base_citation_intent_double_seq_bn_training_adapter_v01":
            #     1==1

            # Check if the number of trials matches the expectation
            if len(trials) > 0 and len(trials) % expected_trials != 0:
                completion_check[study_name] = 'incomplete'
                continue  # Skip further checks as the trial count does not match

            # Check if each trial has the expected number of completed seeds
            if all(seeds_count == expected_seeds_per_trial for seeds_count in trials.values()):
                completion_check[study_name] = 'complete'
            else:
                completion_check[study_name] = 'incomplete'
        
        return completion_check

    def count_completed_seeds(self, study_path):
        """
        For a given study path, count the number of completed seeds within each trial of a study.
        """
        study_data = {}
        study_name = os.path.basename(study_path)

        for trial in os.listdir(study_path):
            trial_path = os.path.join(study_path, trial)
            if not os.path.isdir(trial_path):
                continue
            completed_seeds = 0
            for seed in os.listdir(trial_path):
                seed_path = os.path.join(trial_path, seed)
                if not os.path.isdir(seed_path):
                    continue
                if "trainer_state.json" in os.listdir(seed_path):
                    completed_seeds += 1
            study_data[trial] = completed_seeds

        return {study_name: study_data}

    def get_optuna_completed_studies(self):

        def check_completed_trials(model_type, n_trials):
            if model_type not in self.trials_dict:
                raise ValueError("Invalid model type")
            
            if n_trials > self.trials_dict[model_type] and not n_trials % self.trials_dict[model_type] == 0:
                raise ValueError("Invalid number of trials")
            return n_trials == self.trials_dict[model_type]

        completed_studies = []
        for study_name in self.optuna_studies:
            if study_name.endswith('test') or study_name.endswith('gold'):
                continue
            study = self.load_optuna_study(study_name)
            trial_statuses = set([trial.state.name for trial in study.trials])

            if all(status == 'COMPLETE' for status in trial_statuses):
                # Check number of trials is as expected
                n_trials = len(study.trials)
                model_type = self.get_model_type(study_name)
                completed_trials = check_completed_trials(model_type, n_trials)
                if completed_trials:
                    completed_studies.append(study.study_name)
        return completed_studies

    def load_optuna_study(self, study_name):
        return optuna.study.load_study(study_name, storage=self.storage)

class TrainerAnalytics(TrainerUtilities):
    """
    Only scopes trainer outputs and aims to structure them for analytics
    """
    def __init__(self, trainer_output_path="training_output", storage="sqlite:///db.sqlite3"):
        super().__init__(trainer_output_path, storage)

        self.trainer_output_path = trainer_output_path
        self.study_paths = os.listdir(trainer_output_path)

        self.completed_studies, _ = self.check_study_completion()


        trial_paths_dict = {x: self.get_all_trial_and_seed_paths(os.path.join(self.trainer_output_path, x)) for x in self.completed_studies}

        final_training_entries = []
        final_evaluation_entries = []
        epoch_training_entries = []
        epoch_evaluation_entries = []

        for study_name, trial_dict in trial_paths_dict.items():
            model_type = self.get_model_type(study_name)
            # if model_type == 'gold':
            #     continue
            for trial, seed_paths in trial_dict.items():
                for seed_path in seed_paths:

                    trainer_state_path = os.path.join(seed_path, 'trainer_state.json')

                    # Load the trainer state data
                    trainer_state = self.load_json_data(trainer_state_path)
                    training_entry, evaluation_entry, training_entries, evaluation_entries = self.extract_final_metrics(trainer_state, model_type=='adapter')

                    # Add the study name, trial number and seed number
                    trial_i = int(trial.split('_')[-1])
                    seed_i = int(seed_path.split('_')[-1])

                    training_entry.update({'study': study_name, 'model_type': model_type, 'trial': trial_i,'seed': seed_i})
                    evaluation_entry.update({'study': study_name,'model_type': model_type, 'trial': trial_i,'seed': seed_i})
                    training_entries = [{**entry, 'study': study_name, 'model_type': model_type, 'trial': trial_i, 'seed': seed_i} for entry in training_entries]
                    evaluation_entries = [{**entry, 'study': study_name, 'model_type': model_type, 'trial': trial_i, 'seed': seed_i} for entry in evaluation_entries]
                    
                    final_training_entries.append(training_entry)
                    final_evaluation_entries.append(evaluation_entry)
                    epoch_training_entries.extend(training_entries)
                    epoch_evaluation_entries.extend(evaluation_entries)

        # Create the dataframe
        df_training = pd.DataFrame(final_training_entries)
        df_evaluation = pd.DataFrame(final_evaluation_entries)

        # Now, from the model_name, we need to get the command and extract parameters
        # This is to get the model variant, dataset name, classifier head and config name
        df_training['command'] = df_training['study'].apply(lambda x: self.all_studies_dict[x])
        df_evaluation['command'] = df_evaluation['study'].apply(lambda x: self.all_studies_dict[x])

        df_training[['model_variant', 'dataset_name', 'adapter_config_name', 'config_name', 'version']] = df_training.apply(
            lambda row: pd.Series(self.extract_parameters_from_command(row['command'], row['model_type'])), axis=1)

        df_evaluation[['model_variant', 'dataset_name', 'adapter_config_name', 'config_name', 'version']] = df_evaluation.apply(
            lambda row: pd.Series(self.extract_parameters_from_command(row['command'], row['model_type'])), axis=1)

        from collections import OrderedDict
        abbreviations = OrderedDict([
            ('roberta-base', 'ROBERTA'),
            ('allenai/cs_roberta_base', 'DAPT'),
            ('allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219', 'DAPT_TAPT'),
            ('allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688', 'DAPT_TAPT'),
            ('allenai/dsp_roberta_base_tapt_citation_intent_1688', 'TAPT'),
            ('allenai/dsp_roberta_base_tapt_sciie_3219', 'TAPT'),
            # ('./mlm_model', 'MLM-Base'),
        ])

        # Check unique model_variantsß
        # unique_model_variants = df_training['model_variant'].unique()
        # assert len(unique_model_variants) == len(abbreviations), "Mismatch in model variants and abbreviations"
        # # Find differences
        # set(unique_model_variants) - set(abbreviations.keys())

        df_training['short_study'] = df_training['model_variant'].map(abbreviations)
        df_evaluation['short_study'] = df_evaluation['model_variant'].map(abbreviations)

        # Define the columns to keep and merge on
        common = ['study', 'short_study', 'model_variant', 'model_type', 'dataset_name', 
                      'trial', 'seed', 'epoch', 'adapter_config_name', 'config_name', 'version']
        train_cols = ['train_loss']
        eval_cols = ['eval_loss', 'eval_macro_f1']
        
        # join dataframes
        df = pd.merge(df_training[common + train_cols], df_evaluation[common + eval_cols], on=common, suffixes=('_train', '_eval'))
        
        # drop model_variant = ./mlm_model
        df = df[df['model_variant'] != './mlm_model']

        # Select the trials for which to do the study
        # Ignore first trial for each study, since the model is still adapting
        # df = df[df['trial'] != 0]

        # For model_type = Model, just show first 2 trials as these are run before major adjustments
        # Eval Loss is shown to increase and Train Loss to decrease significantly on higher trials
        # Still unknown reason, but appears Optuna is not correctly trying to minimize the eval_loss
        # Will try to run sequentially rather than in parallel.
        # df = df[~((df['model_type'] == 'model') & (df['trial'] > 2))]

        # Add suffix to short_study based on the dataset_name
        # First, let's map the dataset_name
        dataset_name_map = {
            'citation_intent': 'ACL-ARC',
            'sciie': 'SCIERC'
        }

        # Now create new column, task, with the new value
        df['task'] = df['dataset_name'].map(dataset_name_map)

        # Add suffix of adapter_config_name (upper case) to short_study for adapters
        df['short_study'] = df.apply(lambda x: f"{x['short_study']}_{x['adapter_config_name'].upper()}" if x['model_type'] == 'adapter' else x['short_study'], axis=1)

        # Access example
        # condition = ((df['short_study'] == 'ROBERTA') & (df['version'] == 'v01') & (df['task'] == 'ACL-ARC'))
        # df[condition][['task', 'version', 'short_study','trial', 'eval_macro_f1']].reset_index()
        # df[condition][['task', 'version', 'short_study','trial', 'eval_macro_f1', 'av_trial_macro_f1']].reset_index()

        #  Get the average eval_macro_f1 per trial across seeds
        cols_trial_level = ['study', 'trial']
        df_trial_level = df.groupby(cols_trial_level)['eval_macro_f1'].mean().reset_index()
        df = df.merge(df_trial_level, how='left', on=cols_trial_level, suffixes=('', '_av'))
        df.rename(columns={'eval_macro_f1_av': 'av_trial_macro_f1'}, inplace=True)

        # Get the mean and max f1 average per study
        # Since in the study we aim to find the best hyperparameters, then select the max f1 average per study
        # However, we'll start with just the mean to be conservative
        cols_study_level = ['study']
        df_grouped = df_trial_level.groupby(cols_study_level)['eval_macro_f1'].agg(['mean', 'max']).reset_index()
        df_grouped.rename(columns={'mean': 'av_study_macro_f1', 'max': 'max_study_macro_f1'}, inplace=True)
        df = df.merge(df_grouped, how='left', on=cols_study_level, suffixes=('', '_max'))

        # Save for further study analysis
        os.makedirs('resources', exist_ok=True)
        df.to_csv('resources/df_final_results.csv', index=False)

        # Further processing (to be refacored)
        df['trial_order'] = df['trial']
        df.sort_values(by=['short_study', 'model_variant', 'trial_order'], inplace=True)

        tasks = df['task'].unique()
        for task in tasks:
            versions = df['version'].unique()
            for version in versions:
                if version not in ['v03', 'v04', 'v05', 'v06']:
                    # Skip as these do not update
                    continue
                print("------> Processing task: ", task, "Version: ", version)

                out_dir = f'resources/{task}/{version}'
                os.makedirs(out_dir, exist_ok=True)

                filtered_df = df[(df['task'] == task) & (df['version'] == version)]
                if version != 'v01':
                    # Need to add model results
                    filtered_df = pd.concat([filtered_df, df[(df['task'] == task) & (df['version'] == 'v01')]])
                
                # Get the full list of abbreviations by manually adding adapter abbreviations
                full_abbreviations = list(abbreviations.values()) + ['ROBERTA_SEQ_BN', 'ROBERTA_DOUBLE_SEQ_BN', 'DAPT_SEQ_BN', 'DAPT_DOUBLE_SEQ_BN']
                full_abbreviations = list(dict.fromkeys(full_abbreviations))

                # Make plots
                filters_dict = {'task': task, 'version': version}
                self.plot_losses_by_type(filtered_df, filters_dict, out_dir)

                comparison_results_path = 'comparative_outputs/dont-stop-pretraining-model-outputs.json'
                comparison_results = json.load(open(comparison_results_path))
                comparison_results_task_raw = comparison_results['CS'][task]['performance']
                comparison_results_task = {k: round(v / 100, 4) for k, v in comparison_results_task_raw.items()}

                self.plot_evaluation_f1_macro(filtered_df, filters_dict, full_abbreviations, comparison_results_task, out_dir)

        1 == 1

    def extract_final_metrics(self, data, adapter=False):
        """Extract the last training and evaluation metrics from log history."""
        last_training_entry = last_evaluation_entry = None
        training_entries = []
        evaluation_entries = []

        for entry in reversed(data['log_history']):
            if 'train_loss' in entry and not last_training_entry:
                last_training_entry = entry
            elif 'eval_loss' in entry and not last_evaluation_entry:
                last_evaluation_entry = entry
            # elif adapter and 'loss' in entry and not last_training_entry:
            #     last_training_entry = entry
            else:
                if 'learning_rate' in entry:
                    training_entries.append(entry)
                elif 'eval_loss' in entry:
                    evaluation_entries.append(entry)
                else:
                    raise ValueError("Invalid entry type in log history")

        return last_training_entry, last_evaluation_entry, training_entries, evaluation_entries

    def plot_losses_by_type(self, df, filters_dict, out_dir):

        # Define Adapters as those indexes that have adapter_config_name
        adapter_indexes = df['model_type'] == 'adapter'
        dataframes = {
            'Adapters': df[adapter_indexes],
            'Continued Pretraining': df[~adapter_indexes]
        }
        # Create figures for each type of loss
        for loss_type in ['eval_loss', 'train_loss']:
            plt.figure(figsize=(20, 6))  # Larger figure size for side by side subplots

            n_subplot = 1
            for key, dataframe in dataframes.items():
                if dataframe.empty:
                    continue

                # First subplot for trials in adapters
                plt.subplot(1, 2, n_subplot)
                sns.boxplot(data=dataframe, x='trial', y=loss_type, hue='study')

                filters_subtitle = ', '.join([f'{key}: {value}' for key, value in filters_dict.items()])

                plt.title(f'Boxplot of {loss_type} for {key} \n \
                          {filters_subtitle}')
                plt.xlabel('Trial')
                plt.ylabel(loss_type.capitalize())
                plt.xticks(rotation=45)

                n_subplot += 1

            # if dataframe.empty:
            #     continue

            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.savefig(f'{out_dir}/{loss_type}_across_seeds_comparison.png')
            plt.show()
            print("Finished Loss Type Plot")

    def plot_evaluation_f1_macro(self, df, filters_dict, ordered_labels, comparison_results_task, out_dir):

        # Add benchmarks for adapters
        comparison_results_task.update({
            'ROBERTA_SEQ_BN': df[df['short_study'] == 'TAPT']['av_study_macro_f1'].mean(),
            'ROBERTA_DOUBLE_SEQ_BN': df[df['short_study'] == 'TAPT']['av_study_macro_f1'].mean(),
            'DAPT_SEQ_BN': df[df['short_study'] == 'DAPT_TAPT']['av_study_macro_f1'].mean(),
            'DAPT_DOUBLE_SEQ_BN': df[df['short_study'] == 'DAPT_TAPT']['av_study_macro_f1'].mean()})

        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 8))
        plt.ylim(0.4, 0.85) # Note, consider the annotations if changing these limits
        ax = sns.boxplot(data=df, x='short_study', y='eval_macro_f1', hue='trial', 
            palette='Set2', order=ordered_labels)

        # Font settings for annotations
        fontdict = {'fontsize': 10, 'fontweight': 'bold'}

        # Draw the benchmark lines and the average eval_macro_f1 per study/trial
        for i, study in enumerate(ordered_labels):
            benchmark = comparison_results_task.get(study)
            
            if benchmark:
                # Get the y position for the average eval_macro_f1 per study/trial
                av_f1 = df[df['short_study'] == study]['av_study_macro_f1'].max()

                if study in ['ROBERTA_SEQ_BN', 'ROBERTA_DOUBLE_SEQ_BN', 'DAPT_SEQ_BN', 'DAPT_DOUBLE_SEQ_BN']:
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


        filters_subtitle = ', '.join([f'{key}: {value}' for key, value in filters_dict.items()])
        plt.title(f'Evaluation Macro F1 Score Across Different Seeds for Each Trial \n \
                  {filters_subtitle}')
        plt.xlabel('Study')
        plt.ylabel('Evaluation Macro F1 Score')
        plt.xticks(rotation=45)
        plt.legend(title='Trial', loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f'{out_dir}/evaluation_macro_f1_across_seeds.png', dpi=300)
        plt.show()
        plt.close()
        print("Finished F1 Metric Plot")

    def get_latest_checkpoint_directory(self, seed_path):
        """Return the path to the latest checkpoint directory or the regular trainer state path."""
        checkpoint_dirs = [d for d in os.listdir(seed_path) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            return os.path.join(seed_path, latest_checkpoint, 'trainer_state.json'), True
        return os.path.join(seed_path, 'trainer_state.json'), False

    def load_json_data(self, filepath):
        """Load data from a JSON file."""
        with open(filepath, 'r') as file:
            return json.load(file)

    def extract_study_data(self, study_path):
        """Process each study directory to collect data for DataFrame."""
        study_data = {}
        for trial in os.listdir(study_path):
            trial_path = os.path.join(study_path, trial)
            trial_data = {}
            for seed in os.listdir(trial_path):
                seed_path = os.path.join(trial_path, seed)
                trainer_state_path, adapter = self.get_latest_checkpoint_directory(seed_path)
                if os.path.isfile(trainer_state_path):
                    data = self.load_json_data(trainer_state_path)
                    training_entry, evaluation_entry = self.extract_final_metrics(data, adapter)
                    trial_data[seed] = {
                        'training_entry': training_entry,
                        'evaluation_entry': evaluation_entry
                    }
            study_data[trial] = trial_data
        return study_data

    def create_dataframe_from_data(self, data):
        """Create a DataFrame from structured data."""
        df_data = []
        for study, study_data in data.items():
            for trial, trial_data in study_data.items():
                for seed, seed_entries in trial_data.items():
                    row = {'study': study, 'trial': trial, 'seed': seed, **seed_entries}
                    df_data.append(row)
        return pd.DataFrame(df_data)


class TrainerOutputProcessor(TrainerUtilities):

    def __init__(self, trainer_output_path="training_output", storage="sqlite:///db.sqlite3"):
        super().__init__(trainer_output_path, storage)

        self.completed_studies, remediate_completed = self.check_study_completion()
        self.remediate_studies_dict = self.get_mismatch_storage_studies()
        self.remediate_studies_dict.update({'optuna_failed': self.get_optuna_failed_studies()})
        self.remediate_studies_dict.update({'missing_mlflow': self.get_missing_mlflow_studies()})
        self.remediate_studies_dict.update({'remediate_completed': remediate_completed})
        self.pending_studies = self.get_pending_studies()
        self.pending_studies_commands = [self.all_studies_dict[key] for key in self.pending_studies]

        self.incomplete_studies_dict = self.get_incomplete_studies()
        potentially_halted = set(self.incomplete_studies_dict['trainer_outputs']) - set(self.incomplete_studies_dict['optuna'])
        1==1 # breakpoint area

        # Manual functions for deletion
        # self.delete_folder_and_optuna_study('cs_roberta_base_sciie_seq_bn_training_adapter_v01')
        # self.all_studies_dict['cs_roberta_base_sciie_seq_bn_training_adapter_v01']

        # dsp_roberta_base_dapt_cs_tapt_citation_intent_1688_citation_intent_training_base_v03
        # dsp_roberta_base_tapt_citation_intent_1688_citation_intent_training_base_v03

    def delete_study(self, study_name):
        """
        Delete a study from the training output path and from optuna.
        """
        self.delete_folder_and_optuna_study(self.trainer_output_path, study_name)

    def get_missing_mlflow_studies(self):
        missing_mlflow = set()
        trial_paths_dict = {x: self.get_all_trial_and_seed_paths(os.path.join(self.trainer_output_path, x)) for x in self.completed_studies}
        for study_name, trial_dict in trial_paths_dict.items():
            for trial, seed_paths in trial_dict.items():
                for seed_path in seed_paths:
                    items = os.listdir(seed_path)
                    if 'trainer_state.json' in items and 'mlflow_id.txt' not in items:
                        # print(f"Missing mlflow for {study_name}/{trial}/{seed_path}")
                        missing_mlflow.add(study_name)

                    # # Below some maintenance of checkpoints, to make the results more light-weight
                    # if seed_path == 'training_output/cs_roberta_base_citation_intent_double_seq_bn_training_adapter_v02/trial_0/seed_8937216':
                    #     1==1
                    # # If any item starts with 'checkpoint', then retrieve those folders
                    # checkpoint_dirs = [d for d in items if d.startswith('checkpoint-')]
                    # if checkpoint_dirs:
                    #     for checkpoint_dir in checkpoint_dirs:
                    #         checkpoint_path = os.path.join(seed_path, checkpoint_dir)
                    #         # Delete checkpoint folder
                    #         os.system(f"rm -r {checkpoint_path}")

        # in linux check how large a folder is
        # du -sh folder_name

        #list all subdirectories in skeleton format
        # tree -L 2 -d

        # Those that have checkpoit subfolder
        # find . -type d -name checkpoint

        return missing_mlflow

    def get_mismatch_storage_studies(self):
        """
        Get the studies that are missing from the training output path.
        And the studies that are missing from optuna.
        """
        trainer_outputs_studies = []
        for study in self.optuna_studies:
            if study not in self.study_paths:
                trainer_outputs_studies.append(study)

        optuna_studies = []
        for study in self.study_paths:
            if study not in self.optuna_studies:
                optuna_studies.append(study)

        mismatch_studies = {
            'missing_in_trainer_outputs': trainer_outputs_studies,
            'missing_in_optuna': optuna_studies
        }

        return mismatch_studies

    def get_pending_studies(self):
        # Get the study names from outputs that are not in the in-scope list
        all_study_names = self.all_studies_dict.keys()
        pending_study_names = [study for study in all_study_names if study not in self.study_paths]

        # print the studies that are on optuna
        for study in pending_study_names:
            if study in self.optuna_studies:
                print(f"Pending Study {study} is in optuna.")
                print("Check as you may need to delete it from Optuna")

        return pending_study_names

    def get_commands_for_pending_studies(self, trainer_output_path, datasets, versions):
        bash_commands = []
        for dataset in datasets:
            for version in versions:
                bash_commands.extend(self.container_of_expected_runs(dataset, version))

        all_study_names = [self.extract_study_name_from_cmd(cmd) for cmd in bash_commands]
        study_paths = os.listdir(trainer_output_path)

        # Get the study names that are not in the expected list
        missing_study_names = [study for study in all_study_names if study not in study_paths]

        # Form commands
        for study in missing_study_names:
            command = self.form_command_from_study_name(study)
            print(command)

    def get_incomplete_studies(self):
        """
        Analyses training output dir for studies that are incomplete and could be deleted.
        """
        trainer = self.get_trainer_output_incomplete_studies()
        optuna = self.get_optuna_incomplete_studies()

        both = set(trainer).intersection(set(optuna))
        trainer_only = set(trainer) - set(optuna)
        optuna_only = set(optuna) - set(trainer)

        return {
            'both': both,
            'trainer_outputs': trainer_only,
            'optuna': optuna_only
        }

    def get_trainer_output_incomplete_studies(self):
        all_study_data = {}
        for study_path in self.study_paths:
            all_study_data.update(self.count_completed_seeds(os.path.join(trainer_output_path, study_path)))

        study_completion = self.check_trainer_study_completion(all_study_data)

        # Get the incomplete studies
        incomplete_studies = []
        for study, status in study_completion.items():
            if status == 'incomplete':
                incomplete_studies.append(study)
        return incomplete_studies

    def get_optuna_failed_studies(self):
        failed_studies = []
        for study_name in self.optuna_studies:
            study = self.load_optuna_study(study_name)
            trial_statuses = set([trial.state.name for trial in study.trials])
            if 'FAIL' in trial_statuses:
                failed_studies.append(study_name)
        return failed_studies

    def get_optuna_incomplete_studies(self):
        """ In Development """
        complete_studies = self.get_optuna_completed_studies()
        incomplete_studies = [study for study in self.optuna_studies if study not in complete_studies]
        return incomplete_studies

    def delete_folder_and_optuna_study(self, delete_candidate):
        print(f"Deleting {delete_candidate} folder")
        os.system(f"rm -r {os.path.join(trainer_output_path, delete_candidate)}")

        # Delete from optuna
        from utils import mlops
        storage = "sqlite:///db.sqlite3"
        study_exists = mlops.check_study_exists(storage, delete_candidate)
        if study_exists:
            print(f"Deleting {delete_candidate} from storage.")
            optuna.study.delete_study(delete_candidate, self.storage)
        else:
            print(f"Study {delete_candidate} does not exist in storage.")

if __name__ == "__main__":
    trainer_output_path = "training_output"

    processor = TrainerOutputProcessor(trainer_output_path)
    analytics = TrainerAnalytics(trainer_output_path)
