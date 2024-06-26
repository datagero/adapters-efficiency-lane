"""
These are the utility functions made on-the-go for project demands.
No guarantees of reproduction and are just commited for reference purposes.
Used to process the training output data and generate visualizations.
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

# Generates the LaTeX table for each unique task from the CSV data.
def generate_latex_tables_from_csv(csv_data):
    df = pd.read_csv(csv_data)

    # Function to format a single row in LaTeX syntax
    def format_row(row):
        # Format each cell with value and standard deviation as subscript
        formatted_cells = [row['short_study']] + [f"{row[f'f1_{method}']*100:.2f}$_{{\\text{{ {row[f'std_f1_{method}']*100:.2f}}}}}$"
                           if pd.notna(row[f'f1_{method}']) else ""
                           for method in ['baseline', 'finetuning', 'pfeiffer', 'houlsby']]
        return ' & '.join(formatted_cells)

    # Iterate over each unique task and generate the LaTeX table
    tables = {}
    for task in df['task'].unique():
        # ignore nan
        if pd.isna(task):
            print("Skipping nan task")
            continue
        task_data = df[df['task'] == task]

        # Generate LaTeX table content
        table_content = '\\\\ \n'.join(task_data.apply(format_row, axis=1))
        
        # Define the LaTeX table format
        table_latex = f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{{task}}}
    \\label{{table:{task.lower().replace('-', '')}}}
    \\begin{{tabular}}{{@{{}}lcccc@{{}}}}
    \\toprule
    Pre-trained Model & Baseline & Fine Tuning & Adapter Pfeiffer & Adapter Houlsby \\\\ \\midrule
    {table_content}\\\\
    \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""
        
        # Add the generated table to the dictionary with the task as the key
        tables[task] = table_latex
    
    return tables

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

    def container_of_expected_runs(self, dataset="citation_intent", version="v01", parallelism="1"):

        bash_commands_dict = {
            f"roberta-base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} seq_bn adapter_default adapter_{version}",
            f"roberta-base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} double_seq_bn adapter_default adapter_{version}",
        }

        if dataset in ['citation_intent', 'sciie']:
            bash_commands_dict = {
                f"roberta-base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh roberta-base {dataset} finetuning model_{version}",
                f"roberta-base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} seq_bn adapter_default adapter_{version}",
                f"roberta-base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} double_seq_bn adapter_default adapter_{version}",
                f"cs_roberta_base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/cs_roberta_base {dataset} finetuning model_{version}",
                f"cs_roberta_base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/cs_roberta_base {dataset} seq_bn adapter_default adapter_{version}",
                f"cs_roberta_base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/cs_roberta_base {dataset} double_seq_bn adapter_default adapter_{version}"
            }

            # The best adapters
            bash_commands_dict.update({
                f"roberta-base_{dataset}_seq_bn_training_adapter_{version}_best": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} seq_bn roberta-base_{dataset}_seq_bn adapter_{version}_best",
                f"roberta-base_{dataset}_double_seq_bn_training_adapter_{version}_best": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} double_seq_bn roberta-base_{dataset}_double_seq_bn adapter_{version}_best",
                f"cs_roberta_base_{dataset}_seq_bn_training_adapter_{version}_best": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/cs_roberta_base {dataset} seq_bn cs_roberta_base_{dataset}_seq_bn adapter_{version}_best",
                f"cs_roberta_base_{dataset}_double_seq_bn_training_adapter_{version}_best": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/cs_roberta_base {dataset} double_seq_bn cs_roberta_base_{dataset}_double_seq_bn adapter_{version}_best"
            })

            if dataset == 'citation_intent':
                bash_commands_dict.update({
                    f"mlm_model_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh ./mlm_model {dataset} finetuning model_{version}",
                    f"dsp_roberta_base_tapt_citation_intent_1688_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_tapt_citation_intent_1688 {dataset} finetuning model_{version}",
                    f"dsp_roberta_base_tapt_citation_intent_1688_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_tapt_citation_intent_1688 {dataset} seq_bn adapter_default adapter_{version}",
                    f"dsp_roberta_base_tapt_citation_intent_1688_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_tapt_citation_intent_1688 {dataset} double_seq_bn adapter_default adapter_{version}",
                    f"dsp_roberta_base_dapt_cs_tapt_citation_intent_1688_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 {dataset} finetuning model_{version}",
                    f"dsp_roberta_base_dapt_cs_tapt_citation_intent_1688_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 {dataset} seq_bn adapter_default adapter_{version}",
                    f"dsp_roberta_base_dapt_cs_tapt_citation_intent_1688_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 {dataset} double_seq_bn adapter_default adapter_{version}"
                })
            if dataset == 'sciie':
                bash_commands_dict.update({
                    f"dsp_roberta_base_tapt_sciie_3219_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_tapt_sciie_3219 {dataset} finetuning model_{version}",
                    f"dsp_roberta_base_tapt_sciie_3219_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_tapt_sciie_3219 {dataset} seq_bn adapter_default adapter_{version}",
                    f"dsp_roberta_base_tapt_sciie_3219_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_tapt_sciie_3219 {dataset} double_seq_bn adapter_default adapter_{version}",
                    f"dsp_roberta_base_dapt_cs_tapt_sciie_3219_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219 {dataset} finetuning model_{version}",
                    f"dsp_roberta_base_dapt_cs_tapt_sciie_3219_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219 {dataset} seq_bn adapter_default adapter_{version}",
                    f"dsp_roberta_base_dapt_cs_tapt_sciie_3219_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219 {dataset} double_seq_bn adapter_default adapter_{version}"
                })

        if dataset in ['hyperpartisan_news', 'ag']:

            bash_commands_dict.update({
                f"news_roberta_base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/news_roberta_base {dataset} finetuning model_{version}",
                f"news_roberta_base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/news_roberta_base {dataset} seq_bn adapter_default adapter_{version}",
                f"news_roberta_base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/news_roberta_base {dataset} double_seq_bn adapter_default adapter_{version}",
            })

            if dataset in ['hyperpartisan_news']:
                bash_commands_dict.update({
                    f"roberta-base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh roberta-base {dataset} finetuning model_{version}"
                })

        if dataset in ['amazon', 'imdb']:

            bash_commands_dict.update({
                f"reviews_roberta_base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/reviews_roberta_base {dataset} finetuning model_{version}",
                f"reviews_roberta_base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/reviews_roberta_base {dataset} seq_bn adapter_default adapter_{version}",
                f"reviews_roberta_base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/reviews_roberta_base {dataset} double_seq_bn adapter_default adapter_{version}",
            })

        if dataset in ['chemprot', 'rct-20k']:
            # Need to overwrite base training as it requires different config as it maximises micro-f1 instead of macro-f1
            bash_commands_dict.update({
                f"roberta-base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh roberta-base {dataset} finetuning_biomed model_{version}",
                f"biomed_roberta_base_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/biomed_roberta_base {dataset} finetuning_biomed model_{version}",
                f"roberta-base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} seq_bn adapter_biomed adapter_{version}",
                f"roberta-base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base {dataset} double_seq_bn adapter_biomed adapter_{version}",
                f"biomed_roberta_base_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/biomed_roberta_base {dataset} seq_bn adapter_biomed adapter_{version}",
                f"biomed_roberta_base_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/biomed_roberta_base {dataset} double_seq_bn adapter_biomed adapter_{version}",
            })

            if dataset in ['chemprot']:
                bash_commands_dict.update({
                    f"dsp_roberta_base_tapt_chemprot_4169_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_tapt_chemprot_4169 {dataset} finetuning_biomed model_{version}",
                    f"dsp_roberta_base_dapt_biomed_tapt_chemprot_4169_{dataset}_training_model_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169 {dataset} finetuning_biomed model_{version}",
                    f"dsp_roberta_base_tapt_chemprot_4169_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_tapt_chemprot_4169 {dataset} seq_bn adapter_biomed adapter_{version}",
                    f"dsp_roberta_base_tapt_chemprot_4169_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_tapt_chemprot_4169 {dataset} double_seq_bn adapter_biomed adapter_{version}",
                    f"dsp_roberta_base_dapt_biomed_tapt_chemprot_4169_{dataset}_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169 {dataset} seq_bn adapter_biomed adapter_{version}",
                    f"dsp_roberta_base_dapt_biomed_tapt_chemprot_4169_{dataset}_double_seq_bn_training_adapter_{version}": f"bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169 {dataset} double_seq_bn adapter_biomed adapter_{version}",
                })

        for study, command in bash_commands_dict.items():
            bash_commands_dict[study] = bash_commands_dict[study] + f" {parallelism}"

        return bash_commands_dict

    def extract_parameters_from_command(self, command_string, model_type):
        # Split the command to isolate the relevant sections
        parts = command_string.split()
        
        if model_type == 'model':
            # Extracting for the standard model training
            model_variant = parts[2]
            dataset_name = parts[3]
            config_name = parts[4]
            adapter_config_name = None # ignore to keep consistent
            study_suffix = parts[5]
            version = study_suffix.split('_')[-1]
        elif model_type == 'adapter':
            # Extracting for the adapter training
            model_variant = parts[2]
            dataset_name = parts[3]
            adapter_config_name = parts[4]
            config_name = parts[5]
            study_suffix = parts[6]
            version = study_suffix.split('_')[-1]

        # Order of parameters in the command string
        return model_variant, dataset_name, adapter_config_name, config_name, version


    def get_in_scope_studies(self): 
        study_dict = {}
        for version in ["v01"]:
            # Only scope low-resource datasets
            study_dict.update(self.container_of_expected_runs(dataset="citation_intent", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="sciie", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="hyperpartisan_news", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="chemprot", version=version))

            study_dict.update(self.container_of_expected_runs(dataset="ag", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="amazon", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="imdb", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="rct-20k", version=version))

        for version in ["v01_best"]:
            study_dict.update(self.container_of_expected_runs(dataset="citation_intent", version=version))
            study_dict.update(self.container_of_expected_runs(dataset="sciie", version=version))

        # Models might stop being trained before adapters
        in_scope = {}
        for study_name, command in study_dict.items():
            model_type = self.get_model_type(study_name)
            version = study_name.split('_')[-1]
            if version in ['v04', 'v05', 'v06']:
                if model_type == 'model':
                    continue
                elif model_type == 'adapter' and 'mlm_model' in command:
                    continue

            if study_name.endswith('gold') or study_name.endswith('test') or study_name.endswith('backup'):
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


    def get_successful_trial_and_seed_paths(self, study_name):
        """
        Get all trial and seed paths for a given study path.
        """
        study_path = os.path.join(self.trainer_output_path, study_name)
        all_seed_paths = {}

        study = self.load_optuna_study(study_name)
        trial_statuses = {trial.number: trial.state.name for trial in study.trials}

        for trial in os.listdir(study_path):
            trial_number = int(trial.split('_')[-1])
            trial_status = trial_statuses[trial_number]
            if trial_status != 'COMPLETE':
                continue

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
        elif 'v01_best' in study_name:
            return 'adapter'
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
            if study == 'roberta-base_chemprot_seq_bn_training_adapter_v01':
                1==1
            if status == 'complete':
                complete_studies.append(study)

        optuna_complete_studies = self.get_optuna_completed_studies()

        # check difference
        complete_not_in_optuna = set(complete_studies) - set(optuna_complete_studies)
        optuna_not_in_trainer = set(optuna_complete_studies) - set(complete_studies)

        # add the two to a single set -> To remediate at some point
        remediate_set = complete_not_in_optuna.union(optuna_not_in_trainer)

        complete_studies = [x for x in complete_studies if x in self.all_studies_dict]

        return list(set(complete_studies) - remediate_set), remediate_set

    def check_trainer_study_completion(self, study_data):
        """
        Useful function to identify models that failed / are in progress.
        This info can then be used to clean legacy data or to identify studies that need to be re-run.
        The input is an already processed dictionary with the study names as keys and the number of completed seeds as values.
        """
        completion_check = {}
        
        for study_name, trials in study_data.items():
            if 'v01_best' in study_name:
                # Assume this is completed
                completion_check[study_name] = 'complete'
                continue

            # Using regex to dynamically check for the ending pattern and extract the version number
            model_type = self.get_model_type(study_name)

            if study_name == 'roberta-base_sciie_seq_bn_training_adapter_v01':
                1==1

            if model_type in self.trials_dict:
                # We expect 10 trials for any adapter version and 5 trials for any model version
                expected_trials = self.trials_dict[model_type]
                expected_seeds_per_trial = self.seeds_dict[model_type]
                # if 'sciie' in study_name and model_type == 'adapter':
                #     # We just run two seeds for sciie
                #     expected_seeds_per_trial = 2
            else:
                completion_check[study_name] = 'incomplete'
                continue  # Skip further checks as the study name pattern does not match

            if study_name == "dsp_roberta_base_tapt_sciie_3219_sciie_training_model_v01":
                1==1

            # Check if the number of trials matches the expectation
            if len(trials) > 0 and len(trials) < expected_trials:
                completion_check[study_name] = 'incomplete'
                # To do -> Check optuna to confirm completion of trials
                continue  # Skip further checks as the trial count does not match

            # Check if the study has the minimum required trials with completed seeds
            completed_trials = sum(1 for n_seeds in trials.values() if n_seeds == expected_seeds_per_trial)
            if completed_trials < expected_trials:
                completion_check[study_name] = 'incomplete'
            else:
                completion_check[study_name] = 'complete'

            # if all(seeds_count == expected_seeds_per_trial for seeds_count in trials.values()):
            #     completion_check[study_name] = 'complete'
            # else:
            #     completion_check[study_name] = 'incomplete'
        
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
            
            # if n_trials > self.trials_dict[model_type] and not n_trials % self.trials_dict[model_type] == 0:
            #     raise ValueError("Invalid number of trials")
            return n_trials == self.trials_dict[model_type]

        completed_studies = []
        for study_name in self.optuna_studies:
            if 'v01_best' in study_name:
                # Assume this is completed
                completed_studies.append(study_name)
                continue

            # if 'chemprot' in study_name:
            #     continue
            if study_name.endswith('test') or study_name.endswith('gold') or study_name.endswith('backup'):
                continue
            study = self.load_optuna_study(study_name)
            trial_statuses = set([trial.state.name for trial in study.trials])

            # Count the number of completed trials
            n_completed_trials = len([status for status in [trial.state.name for trial in study.trials] if status == 'COMPLETE'])
            if n_completed_trials >= self.trials_dict[self.get_model_type(study_name)]:
                completed_studies.append(study_name)
            # if all(status == 'COMPLETE' for status in trial_statuses):
            #     # Check number of trials is as expected
            #     n_trials = len(study.trials)
            #     model_type = self.get_model_type(study_name)
            #     completed_trials = check_completed_trials(model_type, n_trials)
            #     if completed_trials:
            #         completed_studies.append(study.study_name)
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

        trial_paths_dict = {x: self.get_successful_trial_and_seed_paths(x) for x in self.completed_studies}
        # trial_paths_dict = {x: self.get_all_trial_and_seed_paths(os.path.join(self.trainer_output_path, x)) for x in self.completed_studies}

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



        # Now, from the epoch_training_entries and epoch_evaluation_entries, let's create some learning curves
        df_epoch_training = pd.DataFrame(epoch_training_entries)
        df_epoch_evaluation = pd.DataFrame(epoch_evaluation_entries)

        df_epoch_training['command'] = df_epoch_training['study'].apply(lambda x: self.all_studies_dict[x])
        df_epoch_evaluation['command'] = df_epoch_evaluation['study'].apply(lambda x: self.all_studies_dict[x])

        df_epoch_training[['model_variant', 'dataset_name', 'adapter_config_name', 'config_name', 'version']] = df_epoch_training.apply(
            lambda row: pd.Series(self.extract_parameters_from_command(row['command'], row['model_type'])), axis=1)

        df_epoch_evaluation[['model_variant', 'dataset_name', 'adapter_config_name', 'config_name', 'version']] = df_epoch_evaluation.apply(
            lambda row: pd.Series(self.extract_parameters_from_command(row['command'], row['model_type'])), axis=1)


        from collections import OrderedDict
        abbreviations = OrderedDict([
            ('roberta-base', 'ROBERTA'),
            ('allenai/cs_roberta_base', 'DAPT'),
            ('allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219', 'DAPT_TAPT'),
            ('allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688', 'DAPT_TAPT'),
            ('allenai/dsp_roberta_base_tapt_citation_intent_1688', 'TAPT'),
            ('allenai/dsp_roberta_base_tapt_sciie_3219', 'TAPT'),
            ('allenai/news_roberta_base', 'DAPT'),
            ('allenai/reviews_roberta_base', 'DAPT'),
            ('allenai/biomed_roberta_base', 'DAPT'),
            ('allenai/dsp_roberta_base_tapt_chemprot_4169', 'TAPT'),
            ('allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169', 'DAPT_TAPT'),
            # ('./mlm_model', 'MLM-Base'),
        ])

        adapter_names = OrderedDict([
            ('seq_bn', 'Pfeiffer'),
            ('double_seq_bn', 'Houlsby'),
        ])

        dataset_name_map = {
            'citation_intent': 'ACL-ARC',
            'sciie': 'SCIERC',
            'hyperpartisan_news': 'HYPERPARTISAN',
            'ag': 'AGNEWS',
            'amazon': 'HELPFULNESS',
            'imdb': 'IMDB',
            'chemprot': 'CHEMPROT',
            'rct-20k': 'RCT'
        }

        # Check unique model_variantsß
        # unique_model_variants = df_training['model_variant'].unique()
        # assert len(unique_model_variants) == len(abbreviations), "Mismatch in model variants and abbreviations"
        # # Find differences
        # set(unique_model_variants) - set(abbreviations.keys())

        dataframes = [df_training, df_evaluation, df_epoch_training, df_epoch_evaluation]
        # Adjust columns as per mappings
        for df in dataframes:
            df['short_study'] = df['model_variant'].map(abbreviations)
            df['adapter_name'] = df['adapter_config_name'].map(adapter_names)
            df['task'] = df['dataset_name'].map(dataset_name_map)

        # Define the columns to keep and merge on
        common = ['study', 'task', 'short_study', 'model_variant', 'model_type', 'dataset_name', 
                      'trial', 'seed', 'epoch', 'adapter_name', 'config_name', 'version']
        train_cols = ['train_loss']
        eval_cols = ['eval_loss', 'eval_macro_f1']
        
        # join dataframes
        df_final = pd.merge(df_training[common + train_cols], df_evaluation[common + eval_cols], on=common, suffixes=('_train', '_eval'))

        # # Now create new column, task, with the new value
        # df['task'] = df['dataset_name'].map(dataset_name_map)


        # For epochs dataframe
        # set(df_epoch_training.columns).intersection(set(df_epoch_evaluation.columns))
        common = ['task', 'study', 'short_study', 'model_variant', 'model_type', 'dataset_name', 
                      'trial', 'seed', 'epoch', 'adapter_name', 'config_name', 'version']
        train_cols = ['loss', 'learning_rate']
        eval_cols = ['eval_macro_f1', 'eval_loss']

        # Merge the dataframes
        df_epoch = pd.merge(df_epoch_training[common + train_cols], df_epoch_evaluation[common + eval_cols], on=common, suffixes=('_train', '_eval'))

        # # Add the short_study column
        # df_epoch['short_study'] = df_epoch['model_variant'].map(abbreviations)
        # # Add the task column
        # df_epoch['task'] = df_epoch['dataset_name'].map(dataset_name_map)
        # # Add the adapter_config_name suffix to short_study for adapters
        # df_epoch['short_study'] = df_epoch.apply(lambda x: f"{x['short_study']}_{x['adapter_config_name'].upper()}" if x['model_type'] == 'adapter' else x['short_study'], axis=1)

        # drop model_variant = ./mlm_model
        df_final = df_final[df_final['model_variant'] != './mlm_model']
        df_epoch = df_epoch[df_epoch['model_variant'] != './mlm_model']


        # Select the trials for which to do the study
        # Ignore first trial for each study, since the model is still adapting
        # df = df[df['trial'] != 0]

        # For model_type = Model, just show first 2 trials as these are run before major adjustments
        # Eval Loss is shown to increase and Train Loss to decrease significantly on higher trials
        # Still unknown reason, but appears Optuna is not correctly trying to minimize the eval_loss
        # Will try to run sequentially rather than in parallel.
        # df = df[~((df['model_type'] == 'model') & (df['trial'] > 2))]


        # Access example
        # condition = ((df['short_study'] == 'ROBERTA') & (df['version'] == 'v01') & (df['task'] == 'ACL-ARC'))
        # df[condition][['task', 'version', 'short_study','trial', 'eval_macro_f1']].reset_index()
        # df[condition][['task', 'version', 'short_study','trial', 'eval_macro_f1', 'av_trial_macro_f1']].reset_index()

        #  Get the average eval_macro_f1 per trial across seeds, and the standard deviation
        cols_trial_level = ['study', 'trial']
        df_trial_level = df_final.groupby(cols_trial_level)['eval_macro_f1'].agg(['max', 'mean', 'std']).reset_index()
        df_trial_level.rename(columns={'max': 'max_trial_macro_f1', 'mean': 'av_trial_macro_f1', 'std': 'std_trial_macro_f1'}, inplace=True)
        df_view = df.merge(df_trial_level, how='left', on=cols_trial_level)


        # Since in the study we aim to find the best hyperparameters, then select the max f1 average per study
        # Find the maximum av_trial_macro_f1 for each study and keep the associated trial number
        base_cols = ['study', 'task', 'model_type', 'short_study', 'adapter_name', 'version', 'trial']

        # Base dataframe
        df_base = df_view[base_cols].drop_duplicates()

        max_trials = df_trial_level.loc[df_trial_level.groupby('study')['av_trial_macro_f1'].idxmax()]
        max_df = df_base.merge(max_trials, on=['study', 'trial'], how='inner')
    
        # drop study col for cleanness
        # max_df.drop('study', axis=1, inplace=True)
        # Order by base_cols
        max_df = max_df.sort_values(by=base_cols[1:])


        performance_table_cols = ['task', 'short_study', 'version']
        metric_cols = ['av_trial_macro_f1', 'std_trial_macro_f1']
        adapter_pfeiffer_view = max_df[max_df['adapter_name'] == 'Pfeiffer']
        adapter_houlsby_view = max_df[max_df['adapter_name'] == 'Houlsby']
        model_view = max_df[max_df['model_type'] == 'model']
        adapter_view = adapter_pfeiffer_view.merge(adapter_houlsby_view, on=performance_table_cols, how='outer', suffixes=('_pfeiffer', '_houlsby'))

        # For the model_view, join back to adapter_view
        performance_view = adapter_view.merge(model_view, on=performance_table_cols, how='outer')
        performance_view.rename(columns={
            'av_trial_macro_f1': 'f1_finetuning', 
            'std_trial_macro_f1': 'std_f1_finetuning',
            'av_trial_macro_f1_pfeiffer': 'f1_pfeiffer',
            'std_trial_macro_f1_pfeiffer': 'std_f1_pfeiffer',
            'av_trial_macro_f1_houlsby': 'f1_houlsby',
            'std_trial_macro_f1_houlsby': 'std_f1_houlsby'},
            inplace=True)
        performance_view = performance_view[performance_table_cols + ['f1_finetuning', 'std_f1_finetuning', 'f1_pfeiffer', 'std_f1_pfeiffer', 'f1_houlsby', 'std_f1_houlsby']]
        
        # The order of short_study should be as listed below
        order = ['ROBERTA', 'TAPT', 'DAPT', 'DAPT_TAPT']
        performance_view['short_study'] = pd.Categorical(performance_view['short_study'], categories=order, ordered=True)
        performance_view.sort_values(by=performance_table_cols, inplace=True)


        domains_mapper = {
            'ACL-ARC': 'CS',
            'SCIERC': 'CS',
            'HYPERPARTISAN': 'NEWS',
            'AGNEWS': 'NEWS',
            'HELPFULNESS': 'REVIEWS',
            'IMDB': 'REVIEWS',
            'CHEMPROT': 'BIOMED',
            'RCT': 'BIOMED'
        }

        # Add domain column
        performance_view['domain'] = performance_view['task'].map(domains_mapper)

        # Load comparative_outputs json
        with open('comparative_outputs/dont-stop-pretraining-model-outputs.json', 'r') as file:
            comparative_outputs = json.load(file)

        # Add results
        for row in performance_view.itertuples():

            domain = row.domain
            task = row.task
            base_model = row.short_study
            f1_finetuning = row.f1_finetuning
            f1_pfeiffer = row.f1_pfeiffer
            f1_houlsby = row.f1_houlsby

            if row.version == 'best':
                base_model = base_model + '_best'

            std_f1_finetuning = row.std_f1_finetuning
            std_f1_pfeiffer = row.std_f1_pfeiffer
            std_f1_houlsby = row.std_f1_houlsby

            performance_entry = comparative_outputs[domain][task]['performance']
            if 'best_finetuning' not in performance_entry:
                performance_entry['best_finetuning'] = {}
            if 'best_pfeiffer' not in performance_entry:
                performance_entry['best_pfeiffer'] = {}
            if 'best_houlsby' not in performance_entry:
                performance_entry['best_houlsby'] = {}

            performance_entry['best_finetuning'][base_model] = {
                'F1': f1_finetuning, 'std_dev': std_f1_finetuning}
            performance_entry['best_pfeiffer'][base_model] = {
                'F1': f1_pfeiffer, 'std_dev': std_f1_pfeiffer}
            performance_entry['best_houlsby'][base_model] = {
                'F1': f1_houlsby, 'std_dev': std_f1_houlsby}
        
        # Save updated json
        with open('comparative_outputs/dont-stop-pretraining-model-outputs_v02.json', 'w') as file:
            json.dump(comparative_outputs, file, indent=4)


        # Load the comparative outputs
        with open('comparative_outputs/dont-stop-pretraining-model-outputs_v02.json', 'r') as file:
            comparative_outputs = json.load(file)
        
        # Structure the data for the latex tables
        rows = []
        for domain, domain_data in comparative_outputs.items():
            for task, task_data in domain_data.items():
                performance_data = task_data['performance']
                for base_model in ['ROBERTA', 'TAPT', 'DAPT', 'DAPT_TAPT', 'ROBERTA_best', 'TAPT_best', 'DAPT_best', 'DAPT_TAPT_best']:
                    if 'best_finetuning' not in performance_data:
                        continue

                    if base_model not in performance_data['best_finetuning']:
                        continue
                    
                    import numpy as np
                    # Create defaults as workaround to support v01_best
                    baseline_data = performance_data['baseline'].get(base_model, {'F1': np.nan, 'std_dev': np.nan})
                    finetuning_data = performance_data['best_finetuning'].get(base_model, {'F1': np.nan, 'std_dev': np.nan})
                    pfeiffer_data = performance_data['best_pfeiffer'].get(base_model, {'F1': np.nan, 'std_dev': np.nan})
                    houlsby_data = performance_data['best_houlsby'].get(base_model, {'F1': np.nan, 'std_dev': np.nan})

                    contents = {
                        'task': task,
                        'short_study': base_model,
                        'f1_baseline': baseline_data['F1'],
                        'std_f1_baseline': baseline_data['std_dev'],
                        'f1_finetuning': finetuning_data['F1'],
                        'std_f1_finetuning': finetuning_data['std_dev'],
                        'f1_pfeiffer': pfeiffer_data['F1'],
                        'std_f1_pfeiffer': pfeiffer_data['std_dev'],
                        'f1_houlsby': houlsby_data['F1'],
                        'std_f1_houlsby': houlsby_data['std_dev']
                    }

                    rows.append(contents)

        comparison_data = pd.DataFrame(rows)
        comparison_data.to_csv('resources/tmp_max_df2.csv', index=False)



        latex_tables = generate_latex_tables_from_csv('resources/tmp_max_df2.csv')
        print(latex_tables['ACL-ARC'].replace('DAPT_TAPT', 'DAPT\\_TAPT').replace('_best', '\\_best'))
        print(latex_tables['SCIERC'].replace('DAPT_TAPT', 'DAPT\\_TAPT').replace('_best', '\\_best'))
        print(latex_tables['HYPERPARTISAN'].replace('DAPT_TAPT', 'DAPT\\_TAPT').replace('_best', '\\_best'))
        print(latex_tables['IMDB'].replace('DAPT_TAPT', 'DAPT\\_TAPT').replace('_best', '\\_best'))
        print(latex_tables['CHEMPROT'].replace('DAPT_TAPT', 'DAPT\\_TAPT').replace('_best', '\\_best'))


        best_epochs = []
        best_trials = []
        hyperparameters = {}
        for row in max_df.itertuples():
            f1_score = row.av_trial_macro_f1
            study = row.study
            trial = row.trial
            # Pick any seed
            seed = df[(df['study'] == study) & (df['trial'] == trial)]['seed'].min()

            # Open folder
            study_path = os.path.join(trainer_output_path, study, f'trial_{trial}', f'seed_{seed}', 'training_args.json')
            training_args = self.load_json_data(study_path)

            # Get learning rate, batch_size and epochs
            learning_rate = training_args['learning_rate']
            batch_size = training_args['per_device_train_batch_size']
            epochs = training_args['num_train_epochs']

            # # Grab best adapter configurations and save them
            # if  ('citation_intent' in study or 'sciie' in study) and (study.startswith('cs_roberta') or study.startswith('roberta-base')):
            #     ('citation_intent' in study and 'intent_seq_bn' in study and 'cs_roberta' in study) or \
            #     ('sciie' in study and 'sciie_seq_bn' in study and 'cs_roberta' in study) or \
            #     ('sciie' in study and 'sciie_seq_bn' in study and 'roberta-base' in study):


            # For best hyperparameters table
            if  ('citation_intent' in study or 'sciie' in study or 'chemprot' in study):

                # if 'adapter_v' in study:
                hyperparameters[study] = {'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs, 'f1_score': f1_score}


                # if 'adapter_v' in study:
                #     print(f"Study: {study}, \
                #         \n ----> F1 Score: {f1_score}, \
                #         \n ----> Trial: {trial}, Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")

                    # # Create a config for the best study
                    # # Take as template the adapter_default.yaml
                    # # Open the adapter_default.yaml
                    # import yaml
                    # study_version = study.split('_')[-1]
                    # with open(f'cs_7643_efficiencylane/training_configs/adapter_{study_version}/adapter_default.yaml', 'r') as file:
                    #     adapter_config = yaml.load(file, Loader=yaml.FullLoader)
                    # # Delete search space
                    # adapter_config.pop('optuna_search_space')
                    # # Place the optimal parameters
                    # adapter_config['training_args']['learning_rate'] = learning_rate
                    # adapter_config['training_args']['per_device_train_batch_size'] = batch_size
                    # adapter_config['training_args']['per_device_eval_batch_size'] = batch_size
                    # adapter_config['training_args']['num_train_epochs'] = epochs
                    
                    # # Only 3 trials to account potential processing variations (we will pick best)
                    # adapter_config['optuna']['n_trials'] = 3

                    # # Save the config
                    # config_name = '_'.join(study.split('_')[:-3])
                    # config_dir = f'cs_7643_efficiencylane/training_configs/adapter_{study_version}_best'
                    # os.makedirs(config_dir, exist_ok=True)
                    # with open(f'{config_dir}/{config_name}.yaml', 'w') as file:
                    #     yaml.dump(adapter_config, file)


            best_trials.append({'study': study, 'trial': trial, 'f1_score': f1_score, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs})
            best_epochs.append(df_epoch[(df_epoch['study'] == study) & (df_epoch['trial'] == trial)])

        # Concatenate the dataframes inside best_epochs
        best_epochs_df = pd.concat(best_epochs)

        # Create dataframe of hyperparameters, the keys should be rows
        df_hyperparameters = pd.DataFrame(hyperparameters).T.reset_index()

        # Join to best_epochs_df
        further_cols = best_epochs_df[base_cols]
        further_cols.drop('trial', axis=1, inplace=True)
        further_cols = further_cols.drop_duplicates()
        df_hyperparameters = df_hyperparameters.merge(further_cols, left_on='index', right_on='study', how='left')

        # # If name contains fusion, set version to fusion
        # df_hyperparameters['version_clean'] = df_hyperparameters['study'].apply(lambda x: 'fusion' if 'fusion' in x else x)
        # df_hyperparameters['version'] = df_hyperparameters['version_clean'].fillna(df_hyperparameters['version'])

        df_hyperparameters.drop('index', axis=1, inplace=True)
        df_hyperparameters.drop('study', axis=1, inplace=True)

        # Only keep certain versions
        df_hyperparameters = df_hyperparameters[df_hyperparameters['version'].isin(['v01', 'fusion'])]
        df_hyperparameters.to_csv('resources/best_hyperparameters.csv', index=False)

        # Print it with separator &
        df_hyperparameters.sort_values(by=['short_study', 'task', 'adapter_name'], inplace=True, ascending=False)

        # Ignore where adapter_name is NaN
        df_hyperparameters = df_hyperparameters[~df_hyperparameters['adapter_name'].isna()]

        latex_cols = ['short_study', 'adapter_name', 'learning_rate', 'batch_size', 'epochs']
        for task in df_hyperparameters['task'].unique():
            print(task)
            data = df_hyperparameters[df_hyperparameters['task'] == task][latex_cols]
            for row in data.itertuples():
                formatted_cells = [row.short_study.replace('_', '\\_'), row.adapter_name, f"{row.learning_rate:.6f}", int(row.batch_size), int(row.epochs)]
                print(' & '.join(map(str, formatted_cells)) + ' \\\\')
            print('\\hline')


        # order by model_type, sh


        # For each short_study, create a plot of the learning curves for each seed across epochs
        # Unique short studies
        short_studies = best_epochs_df['short_study'].unique()

        # Set up the plot style
        # Set the aesthetic style of the plots
        sns.set(style="whitegrid")






        # Manual pick
        # df[df['study'] == 'roberta-base_citation_intent_seq_bn_training_adapter_v01'][
        #     'short_study', 'model_variant', 'model_type', 'dataset_name', 
        #     'trial', 'seed', 'epoch', 'adapter_config_name', 'config_name', 
        #     'version', 'train_loss', 'eval_loss', 'eval_macro_f1', 'task', 'av_trial_macro_f1']

        metrics = ['max_trial_macro_f1', 'av_trial_macro_f1', 'std_trial_macro_f1']
        df_final = df_final.merge(max_df[['study'] + metrics], on='study', how='left')

        # Save for further study analysis
        os.makedirs('resources', exist_ok=True)
        df_final.to_csv('resources/df_final_results.csv', index=False)

        # Now, retrieve the best trial parameters from trainin_output folder





        df_final['study_adapter'] = df_final.apply(lambda row: row['short_study'] + '_' + row['adapter_name'] if not pd.isna(row['adapter_name']) else row['short_study'], axis=1)

        # Further processing (to be refacored)
        df_final['trial_order'] = df_final['trial']
        df_final.sort_values(by=['study_adapter', 'model_variant', 'trial_order'], inplace=True)

        df_final['domain'] = df_final['task'].map(domains_mapper)


        tasks = df_final['task'].unique()
        for task in tasks:
            versions = df_final['version'].unique()
            for version in versions:
                # if version not in ['v03', 'v04', 'v05', 'v06']:
                #     # Skip as these do not update
                #     continue
                print("------> Processing task: ", task, "Version: ", version)

                # Ignore best version
                if version == 'best':
                    continue
    
                out_dir = f'resources/{task}/{version}'
                os.makedirs(out_dir, exist_ok=True)

                filtered_df = df_final[(df_final['task'] == task) & (df_final['version'] == version)]
                domain = filtered_df['domain'].iloc[0]
                if version != 'v01':
                    # Need to add model results
                    filtered_df = pd.concat([filtered_df, df_final[(df_final['task'] == task) & (df_final['version'] == 'v01')]])
                
                # Get the full list of abbreviations by manually adding adapter abbreviations
                full_abbreviations = list(abbreviations.values()) + ['ROBERTA_Pfeiffer', 'ROBERTA_Houlsby', 'DAPT_Pfeiffer', 'DAPT_Houlsby', 'DAPT_TAPT_Pfeiffer', 'DAPT_TAPT_Houlsby']
                full_abbreviations = list(dict.fromkeys(full_abbreviations))



                # ================== Model & Adapter Plots ==================

                # Plot Learning Curves
                model_types = best_epochs_df['model_type'].unique()
                for model_type in model_types:
                    model_data = best_epochs_df[(best_epochs_df['model_type'] == model_type) & (best_epochs_df['task'] == task) & (best_epochs_df['version'] == version)]
                    self.plot_learning_curves(model_data, model_type, out_dir)

                # Plot the evaluation metrics against benchmarks
                filters_dict = {'task': task, 'version': version}
                self.plot_losses_by_type(filtered_df, filters_dict, out_dir)

                comparison_results_path = 'comparative_outputs/dont-stop-pretraining-model-outputs.json'
                comparison_results = json.load(open(comparison_results_path))
                comparison_results_task_raw = comparison_results[domain][task]['performance']['baseline']
                comparison_results_task = {k: round(v['F1'] / 100, 4) for k, v in comparison_results_task_raw.items()}
                self.plot_evaluation_f1_macro(filtered_df, filters_dict, full_abbreviations, comparison_results_task, out_dir)

                # ================== Adapter Exclusive Plots, for project report ==================
                # Only adapter plots

                base_adapter = 'ROBERTA_Houlsby'

                report_output_dir = f'project_report/resources/{task}/{version}'
                os.makedirs(report_output_dir, exist_ok=True)

                scoped_df = filtered_df[filtered_df['model_type'] == 'adapter']
                scoped_df = scoped_df[scoped_df['short_study'] == 'ROBERTA']
                scoped_df = scoped_df[scoped_df['study_adapter'] == base_adapter]
                self.plot_losses_by_type(scoped_df, filters_dict, report_output_dir)
                
                # adapter_abbreviations = ['ROBERTA_Houlsby', 'DAPT_Houlsby']
                adapter_abbreviations = [base_adapter]
                self.plot_evaluation_f1_macro(filtered_df, filters_dict, adapter_abbreviations, comparison_results_task, report_output_dir)

                report_output_dir_dapttapt = f'{report_output_dir}/DAPT_TAPT'
                os.makedirs(report_output_dir_dapttapt, exist_ok=True)
                adapter_abbreviations = ['DAPT_TAPT_Pfeiffer', 'DAPT_TAPT_Houlsby']
                adapter_abbreviations = list(dict.fromkeys(adapter_abbreviations))
                self.plot_evaluation_f1_macro(filtered_df, filters_dict, adapter_abbreviations, comparison_results_task, report_output_dir_dapttapt)

                # Plot Learning Curves
                adapter_best_epoch_data = best_epochs_df[(best_epochs_df['model_type'] == 'adapter') & (best_epochs_df['task'] == task) & (best_epochs_df['version'] == version)]
                adapter_best_epoch_data = adapter_best_epoch_data[(adapter_best_epoch_data['short_study'] == 'ROBERTA') & (adapter_best_epoch_data['adapter_name'] == 'Pfeiffer')]
                self.plot_learning_curves(adapter_best_epoch_data, 'adapter', report_output_dir)


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

    def plot_learning_curves(self, df, model_type, out_dir):
        # Filter data for the current model type
        model_data = df[(df['model_type'] == model_type)]

        if model_data.empty:
            return

        # Get unique studies and adapter_name combination for this model type
        if model_type == 'adapter':
            model_data['study_adapter'] = model_data['short_study'] + '_' + model_data['adapter_name']
        else:
            model_data['study_adapter'] = model_data['short_study']

        studies_adapter = model_data['study_adapter'].drop_duplicates()
        
        # Calculate the number of rows needed for the subplots
        n_studies = len(studies_adapter)
        n_rows = (n_studies + 1) // 2  # Ensure enough rows for all studies
        
        def make_ax(ax, study, font_size):

            # Filter data for the current study
            study_data = model_data[model_data['study_adapter'] == study]
            # Ignore best version run as not needed
            study_data = study_data[~study_data['version'].str.contains('best')]

            trial = set(study_data['trial'])
            assert len(trial) == 1, "Multiple trials found for the same best study"

            # Plot setup for the subplot
            ax.set_title(f'Best Triall #{trial} Learning Curves for {study} ({model_type})', fontsize=font_size+3)
            ax.set_xlabel('Epoch', fontsize=font_size)
            ax.set_ylabel('Eval Macro F1', fontsize=font_size)

            # Plot learning curve for each seed
            for seed in study_data['seed'].unique():
                subset = study_data[study_data['seed'] == seed]
                sns.lineplot(x='epoch', y='eval_macro_f1', data=subset, label=f'Seed {seed}', ax=ax)

            ax.legend(title='Seeds', fontsize=font_size-1)


        # Create a figure with subplots in a two-column layout
        if n_studies > 1:
            fig, axs = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows), sharex=True)
            font_size = 12
        else:
            fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
            font_size = 16

        # Flatten the axis array and trim any excess if the number of studies is odd
        if n_studies > 1:
            axs = axs.flatten()

            for ax in axs[n_studies:]:  # Hide any unused axes
                ax.set_visible(False)

            # Iterate over studies to create each subplot
            for ax, study in zip(axs, studies_adapter):
                make_ax(ax, study, font_size)
        else:
            make_ax(axs, studies_adapter.iloc[0], font_size)

        # Adjust layout
        tasks = list(set(model_data['task']))
        assert len(tasks) == 1, "Multiple tasks found for the same best study"
        plt.suptitle(f'Learning Curves Overview - {model_type}\nTask - {tasks[0]}', fontsize=font_size+3, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{out_dir}/learning_curves_{model_type}.png')
        # plt.show()
        plt.close()


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
            # plt.show()
            plt.close()
            print("Finished Loss Type Plot")

    def plot_evaluation_f1_macro(self, df, filters_dict, ordered_labels, comparison_results_task, out_dir):

        # Add benchmarks for adapters
        comparison_results_task.update({
            'ROBERTA_Pfeiffer': df[df['study_adapter'] == 'TAPT']['av_trial_macro_f1'].mean(),
            'ROBERTA_Houlsby': df[df['study_adapter'] == 'TAPT']['av_trial_macro_f1'].mean(),
            'DAPT_Pfeiffer': df[df['study_adapter'] == 'DAPT_TAPT']['av_trial_macro_f1'].mean(),
            'DAPT_Houlsby': df[df['study_adapter'] == 'DAPT_TAPT']['av_trial_macro_f1'].mean(),
            'DAPT_TAPT_Pfeiffer': df[df['study_adapter'] == 'DAPT_TAPT']['av_trial_macro_f1'].mean(),
            'DAPT_TAPT_Houlsby': df[df['study_adapter'] == 'DAPT_TAPT']['av_trial_macro_f1'].mean()})

        # Set the aesthetic style of the plots
        if len(ordered_labels) == 1:
            width = 8
            set_min_y = 0
            font_size = 16
        else:
            width = 14
            set_min_y = 9999
            font_size = 14

        sns.set_style("whitegrid")
        plt.figure(figsize=(width, 8))

        if filters_dict['task'] == 'ACL-ARC':
            min_y, max_y = min(set_min_y, 0.4), 0.85
        elif filters_dict['task'] == 'SCIERC':
            min_y, max_y = min(set_min_y, 0.6), 0.95
        elif filters_dict['task'] == 'HYPERPARTISAN':
            min_y, max_y = min(set_min_y, 0.7), 1
        elif filters_dict['task'] == 'AGNEWS':
            min_y, max_y = min(set_min_y, 0.8), 1
        elif filters_dict['task'] == 'HELPFULNESS':
            min_y, max_y = min(set_min_y, 0.4), 0.85
        elif filters_dict['task'] == 'IMDB':
            min_y, max_y = min(set_min_y, 0.8), 1
        elif filters_dict['task'] == 'CHEMPROT':
            min_y, max_y = min(set_min_y, 0.6), 0.95
        elif filters_dict['task'] == 'RCT':
            min_y, max_y = min(set_min_y, 0.6), 0.95
        else:
            raise ValueError("Invalid task")

        plt.ylim(min_y, max_y) # Note, consider the annotations if changing these limits

        # from pandas.api.types import CategoricalDtype
        # cat_type = CategoricalDtype(categories=ordered_labels, ordered=True)
        # df['study_adapter'] = df['study_adapter'].astype(cat_type)

        try:
            ax = sns.boxplot(data=df, x='study_adapter', y='eval_macro_f1', hue='trial', palette='Set2',
                         order=ordered_labels)
        except Exception as e:
            print("Error in boxplot", e)
            ax = sns.boxplot(data=df, x='study_adapter', y='eval_macro_f1', hue='trial', palette='Set2')

        # Draw the benchmark lines and the average eval_macro_f1 per study/trial
        for i, study in enumerate(ordered_labels):
            benchmark = comparison_results_task.get(study)
            
            if benchmark:
                # Get the y position for the average eval_macro_f1 per study/trial
                av_f1 = df[df['study_adapter'] == study]['av_trial_macro_f1'].max()

                if study in ['ROBERTA_Pfeiffer', 'ROBERTA_Houlsby', 'DAPT_Pfeiffer', 'DAPT_Houlsby', 'DAPT_TAPT_Pfeiffer', 'DAPT_TAPT_Houlsby']:
                    # These are experiments for hyperparameter tuning, so we take the best results per trial (trial is made up of 5 seeds)
                    av_f1 = df[df['study_adapter'] == study]['av_trial_macro_f1'].max()

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
                plt.annotate(f'{benchmark:.2f}', (x_position, benchmark + (0.025)*2), textcoords="offset points", xytext=(0,10),
                            ha='center', va='bottom', color='red', fontweight='bold', fontsize=font_size-2,
                            arrowprops=dict(arrowstyle=benchmark_arrow_style, color='red'))

                # Annotate the average eval_macro_f1
                plt.annotate(f'{av_f1:.2f}', (x_position, benchmark + 0.025), textcoords="offset points", xytext=(0,-15),
                            ha='center', va='top', color='green', fontweight='bold', fontsize=font_size-2,
                            arrowprops=dict(arrowstyle=av_f1_arrow_style, color='green'))


        benchmark_patch = mpatches.Patch(color='red', label='TAPT Benchmark', linestyle='--', linewidth=2)
        av_f1_patch = mpatches.Patch(color='green', label='Average eval_macro_f1', linestyle='-', linewidth=2)

        # Adding legends to the plot
        legend1 = ax.legend(handles=[benchmark_patch, av_f1_patch], loc='lower right', bbox_to_anchor=(1, 0), title="Metrics", fontsize=font_size-2)
        ax.add_artist(legend1)  # Adding the first legend manually so it doesn't disappear when adding the second legend for trials


        filters_subtitle = ', '.join([f'{key}: {value}' for key, value in filters_dict.items()])
        plt.title(f'Evaluation Macro F1 Score Across Different Seeds for Each Trial \n \
                  {filters_subtitle}', fontsize=font_size+3)
        plt.xlabel('Study')
        plt.ylabel('Evaluation Macro F1 Score', fontsize=font_size)
        if len(ordered_labels) > 1:
            plt.xticks(rotation=45)
        plt.legend(title='Trial', loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size-2)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/evaluation_macro_f1_across_seeds.png', dpi=300)
        # plt.show()
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
        self.remediate_studies_dict.update({'optuna_failed': self.get_optuna_failed_studies()[0]})
        self.remediate_studies_dict.update({'optuna_contains_failed': self.get_optuna_failed_studies()[1]})
        self.remediate_studies_dict.update({'missing_mlflow': self.get_missing_mlflow_studies()})
        self.remediate_studies_dict.update({'remediate_completed': remediate_completed})
        self.pending_studies = self.get_pending_studies()
        self.pending_studies_commands = [self.all_studies_dict[key] for key in self.pending_studies]

        self.incomplete_studies_dict = self.get_incomplete_studies()
        potentially_halted = set(self.incomplete_studies_dict['trainer_outputs']) - set(self.incomplete_studies_dict['optuna'])
        1==1 # breakpoint area

        confirmed_run = [
            'cs_roberta_base_citation_intent_seq_bn_training_adapter_v01', 
            'cs_roberta_base_citation_intent_double_seq_bn_training_adapter_v01']


        [print(x) for x in self.pending_studies_commands]
        [print(command + " 0") for name, command in self.all_studies_dict.items() if 'chemprot' in name and 'dsp_roberta' not in name]

        # hyperpartisan = [x for x in self.all_studies_dict if 'hyperpartisan' in x]
        # [self.delete_folder_and_optuna_study(x) for x in hyperpartisan]

        # print("Pending Studies ---------------")
        # [print(x) for x in self.pending_studies]

        # print("Remediate Studies ---------------")
        # for key, items in self.remediate_studies_dict.items():
        #     print("======> ", key)
        #     [print(x) for x in items]

        # print("Incomplete Studies ---------------")
        # for key, items in self.incomplete_studies_dict.items():
        #     print("======> ", key)
        #     [print(x) for x in items]

        # Manual functions for deletion
        # self.delete_folder_and_optuna_study('cs_roberta_base_sciie_double_seq_bn_training_adapter_v01-backup')
        # self.delete_folder_and_optuna_study('roberta-base_sciie_seq_bn_training_adapter_v01_backup')
        # self.delete_folder_and_optuna_study('roberta-base_ag_seq_bn_training_default_test')
        # self.delete_folder_and_optuna_study('roberta-base_sciie_seq_bn_training_adapter_v01')
        # self.delete_folder_and_optuna_study('roberta-base_chemprot_seq_bn_training_default_test')

        # # # Rerun
        [print(self.all_studies_dict[x]  + " 0") for x in self.incomplete_studies_dict['not_updated_in_last_hour'] if x in self.all_studies_dict]
        # print(self.all_studies_dict['roberta-base_sciie_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['biomed_roberta_base_chemprot_double_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['roberta-base_ag_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['dsp_roberta_base_tapt_sciie_3219_sciie_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['reviews_roberta_base_imdb_training_model_v01']  + " 0")
        # print(self.all_studies_dict['roberta-base_amazon_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['reviews_roberta_base_amazon_training_model_v01']  + " 0")
        # print(self.all_studies_dict['roberta-base_ag_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['roberta-base_ag_double_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['roberta-base_amazon_double_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['news_roberta_base_ag_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['roberta-base_imdb_double_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['news_roberta_base_ag_double_seq_bn_training_adapter_v01']  + " 0")
        # print(self.all_studies_dict['reviews_roberta_base_imdb_seq_bn_training_adapter_v01']  + " 0")


        # Overwrite
        # print(self.all_studies_dict[''] + " 1"



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

        # Check which optuna studies have not updated a trial in the last hour
        optuna_studies_dict = {study: self.load_optuna_study(study) for study in optuna}
        optuna_studies_last_updated = {study_name: max([trial.datetime_start for trial in study.trials]) for study_name, study in optuna_studies_dict.items()}
        optuna_studies_last_updated = {k: v for k, v in sorted(optuna_studies_last_updated.items(), key=lambda item: item[1])}

        # List the ones not updated in the last hour
        from datetime import datetime
        not_recently_updated = []
        for study, last_updated in optuna_studies_last_updated.items():
            if (datetime.now() - last_updated).seconds > 60*60:
                not_recently_updated.append(study)

        both = set(trainer).intersection(set(optuna))
        trainer_only = set(trainer) - set(optuna)
        optuna_only = set(optuna) - set(trainer)

        return {
            'both': both,
            'trainer_outputs': trainer_only,
            'optuna': optuna_only,
            'not_updated_in_last_hour': not_recently_updated
        }

    def get_trainer_output_incomplete_studies(self):
        all_study_data = {}
        for study_path in self.study_paths:
            all_study_data.update(self.count_completed_seeds(os.path.join(trainer_output_path, study_path)))

        study_completion = self.check_trainer_study_completion(all_study_data)

        # Get the incomplete studies
        incomplete_studies = []
        for study, status in study_completion.items():
            if study not in self.all_studies_dict.keys():
                continue
            if status == 'incomplete':
                incomplete_studies.append(study)
        return incomplete_studies

    def get_optuna_failed_studies(self):
        failed_studies = []
        contains_failed_studies = []
        complete_studies = self.get_optuna_completed_studies()
        for study_name in self.optuna_studies:
            study = self.load_optuna_study(study_name)
            trial_statuses = set([trial.state.name for trial in study.trials])
            if 'FAIL' in trial_statuses:
                if study_name not in complete_studies:
                    # If the study contains a fail and has not completed
                    failed_studies.append(study_name)
                contains_failed_studies.append(study_name)
        return failed_studies, contains_failed_studies

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

    # processor = TrainerOutputProcessor(trainer_output_path)
    analytics = TrainerAnalytics(trainer_output_path)
