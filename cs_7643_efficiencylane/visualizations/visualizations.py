import os
from utils_trainer_path import extract_study_data, create_dataframe_from_data

training_outputs_path = 'training_output'
all_data = {}
for study in os.listdir(training_outputs_path):
    study_path = os.path.join(training_outputs_path, study)
    all_data[study] = extract_study_data(study_path)

df = create_dataframe_from_data(all_data)
1==1