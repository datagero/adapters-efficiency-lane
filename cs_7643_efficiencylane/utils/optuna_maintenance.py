# Set of scripts for the maintenance of Optuna database

import optuna

storage = "sqlite:///db.sqlite3"
studies = optuna.get_all_study_names(storage)

delete_scope = [x for x in studies if x.startswith('cs_roberta_base_training')]

for study in delete_scope:
    optuna.delete_study(study_name=study, storage=storage)

1==1