from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
import adapters.composition as ac
import adapters
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig, RobertaForSequenceClassification
import torch

# Our built utilities
from utils import yaml_utils, compute_metrics #mlops,
from data_loaders.citation_intent_data_loader import CSTasksDataLoader
from demo.adapter_fusion_utils import AdapterCreator, FusionCreator


seeds = [42, 128, 9091, 746483, 8937216]

# for seed in seeds: 
#     fusion_creator = FusionCreator(
#         model_type = 'base',
#         model_name = 'roberta-base',
#         dataset_name = 'citation_intent',
#         seed = seed,
#         ad_lr=1.5e-5,
#         warmup_steps=0,
#         adapter1_name = 'adapter_base_sciie',
#         adapter2_name = 'adapter_base_sciie',
#         batch_size = 16,
#         epochs = 12,
#         trial=5
#     )
#     fusion_model = fusion_creator.create_and_save_adapter()

# for seed in seeds:
#     fusion_creator = FusionCreator(
#         model_type = 'cs',
#         model_name = 'allenai/cs_roberta_base',
#         dataset_name = 'citation_intent',
#         seed = seed,
#         ad_lr=1e-4,
#         warmup_steps=0,
#         adapter1_name = 'adapter_cs_sciie',
#         adapter2_name = 'adapter_cs_sciie',
#         batch_size = 32,
#         epochs =16,
#         trial=2
#     )
#     fusion_model = fusion_creator.create_and_save_adapter()

# seeds = [42, 128, 9091, 746483, 8937216]
# for seed in seeds:
#     fusion_creator = FusionCreator(
#         model_type = 'base',
#         model_name = 'roberta-base',
#         dataset_name = 'sciie',
#         seed = seed,
#         ad_lr=5e-6,
#         warmup_steps=0,
#         adapter1_name = 'adapter_base_citation_intent',
#         adapter2_name = 'adapter_base_citation_intent',
#         batch_size = 16,
#         epochs=10,
#         trial=1
#     )
#     fusion_model = fusion_creator.create_and_save_adapter()

# seeds = [42]#128, 9091, 746483, 8937216]
# for seed in seeds:
#     fusion_creator = FusionCreator(
#         model_type = 'base',
#         model_name = 'roberta-base',
#         dataset_name = 'sciie',
#         seed = seed,
#         ad_lr=5e-6,
#         warmup_steps=0,
#         adapter1_name = 'adapter_base_citation_intent',
#         adapter2_name = 'adapter_base_citation_intent',
#         batch_size = 16,
#         epochs=10,
#         trial=1
#     )
#     fusion_model = fusion_creator.create_and_save_adapter()
# seeds = [42, 128, 9091, 746483, 8937216]
# for seed in seeds:
#     fusion_creator = FusionCreator(
#         model_type = 'cs',
#         model_name = 'allenai/cs_roberta_base',
#         dataset_name = 'sciie',
#         seed = seed,
#         ad_lr=5e-6,
#         warmup_steps=0,
#         adapter1_name = 'adapter_cs_citation_intent',
#         adapter2_name = 'adapter_cs_sciie',
#         batch_size = 16,
#         epochs =12,
#         trial=2
#     )
#     fusion_model = fusion_creator.create_and_save_adapter()

# #FIRST ONE CI BASE
# learning_rates = [5e-6, 5e-5, 1e-4]
# batch_sizes = [16]
# epochs_set = [12]
# trial = 1
# for learning_rate in learning_rates:
#     for i, batch_size in enumerate(batch_sizes):
#         trial += 1
#         fusion_creator = FusionCreator(
#             model_type = 'base',
#             model_name = 'roberta-base',
#             dataset_name = 'citation_intent',
#             seed = 1,
#             ad_lr=learning_rate,
#             warmup_steps=0,
#             adapter1_name = 'adapter_base_sciie',
#             adapter2_name = 'adapter_base_sciie',
#             batch_size = batch_size,
#             epochs =epochs_set[i],
#             trial=trial
#         )
#         fusion_model = fusion_creator.create_and_save_adapter()

# learning_rates = [5e-6, 5e-5, 1e-4]
# batch_sizes = [16]
# epochs_set = [12]
# trial = 1
# for learning_rate in learning_rates:
#     for i, batch_size in enumerate(batch_sizes):
#         trial += 1
#         fusion_creator = FusionCreator(
#             model_type = 'cs',
#             model_name = 'allenai/cs_roberta_base',
#             dataset_name = 'citation_intent',
#             seed = 1,
#             ad_lr=learning_rate,
#             warmup_steps=0,
#             adapter1_name = 'adapter_cs_sciie',
#             adapter2_name = 'adapter_cs_sciie',
#             batch_size = batch_size,
#             epochs =epochs_set[i],
#             trial=trial
#         )
#         fusion_model = fusion_creator.create_and_save_adapter()

# learning_rates = [5e-6, 5e-5, 1e-4]
# batch_sizes = [16]
# epochs_set = [12]
# trial = 1
# for learning_rate in learning_rates:
#     for i, batch_size in enumerate(batch_sizes):
#         trial += 1
#         fusion_creator = FusionCreator(
#             model_type = 'base',
#             model_name = 'roberta-base',
#             dataset_name = 'sciie',
#             seed = 1,
#             ad_lr=learning_rate,
#             warmup_steps=0,
#             adapter1_name = 'adapter_base_citation_intent',
#             adapter2_name = 'adapter_base_citation_intent',
#             batch_size = batch_size,
#             epochs =epochs_set[i],
#             trial=trial
#         )
#         fusion_model = fusion_creator.create_and_save_adapter()

learning_rates = [5e-6, 5e-5, 1.5e-4]
batch_sizes = [16]
epochs_set = [12]
trial = 1
for learning_rate in learning_rates:
    for i, batch_size in enumerate(batch_sizes):
        trial += 1
        fusion_creator = FusionCreator(
            model_type = 'cs',
            model_name = 'allenai/cs_roberta_base',
            dataset_name = 'sciie',
            seed = 1,
            ad_lr=learning_rate,
            warmup_steps=0,
            adapter1_name = 'adapter_cs_citation_intent',
            adapter2_name = 'adapter_cs_citation_intent',
            batch_size = batch_size,
            epochs =epochs_set[i],
            trial=trial
        )
        fusion_model = fusion_creator.create_and_save_adapter()

# learning_rates = [5e-6, 5e-5, 1e-4]
# batch_sizes = [16]#, 32, 64]
# epochs_set = [12]#, 16, 20]
# trial = 1
# for learning_rate in learning_rates:
#     for i, batch_size in enumerate(batch_sizes):
#         trial += 1
#         fusion_creator = FusionCreator(
#             model_type = 'base',
#             model_name = 'roberta-base',
#             dataset_name = 'citation_intent',
#             seed = 1,
#             ad_lr=learning_rate,
#             warmup_steps=0,
#             adapter1_name = 'adapter_cs_sciie',
#             adapter2_name = 'adapter_cs_sciie',
#             batch_size = batch_size,
#             epochs =epochs_set[i],
#             trial=trial
#         )
#         fusion_model = fusion_creator.create_and_save_adapter()


# learning_rates = [5e-6, 1e-5, 5e-5, 1e-4]
# batch_sizes = [16, 32, 64]
# epochs_set = [12, 16, 20]
# trial = 1
# for learning_rate in learning_rates:
#     for i, batch_size in enumerate(batch_sizes):
#         trial += 1
#         fusion_creator = FusionCreator(
#             model_type = 'cs',
#             model_name = 'allenai/cs_roberta_base',
#             dataset_name = 'sciie',
#             seed = 1,
#             ad_lr=learning_rate,
#             warmup_steps=0,
#             adapter1_name = 'adapter_cs_citation_intent',
#             adapter2_name = 'adapter_cs_sciie',
#             batch_size = batch_size,
#             epochs =epochs_set[i],
#             trial=trial
#         )
#         fusion_model = fusion_creator.create_and_save_adapter()

# learning_rates = [5e-5, 1e-4]
# batch_sizes = [16]
# epochs_set = [10]
# trial = 4
# for learning_rate in learning_rates:
#     for i, batch_size in enumerate(batch_sizes):
#         trial += 1
#         fusion_creator = FusionCreator(
#             model_type = 'base',
#             model_name = 'roberta-base',
#             dataset_name = 'sciie',
#             seed = 1,
#             ad_lr=learning_rate,
#             warmup_steps=0,
#             adapter1_name = 'adapter_base_citation_intent',
#             adapter2_name = 'adapter_base_citation_intent',
#             batch_size = batch_size,
#             epochs=epochs_set[i],
#             trial=trial
#         )
#         fusion_model = fusion_creator.create_and_save_adapter()

