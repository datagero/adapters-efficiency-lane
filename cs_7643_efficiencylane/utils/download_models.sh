out_dir="/Users/datagero/Documents/scratch/cs-7643-projectfiles"
source_dir="/Users/datagero/Documents/scratch/dont-stop-pretraining"

# # Create the output directory if it doesn't exist
# mkdir -p "$output_dir"

# Change directory to the source directory
cd "$source_dir" || exit

python -m scripts.download_model \
        --model allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 \
        --serialization_dir ${out_dir}/pretrained_models/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688

python -m scripts.download_model \
        --model allenai/dsp_roberta_base_tapt_citation_intent_1688 \
        --serialization_dir ${out_dir}/pretrained_models/dsp_roberta_base_tapt_citation_intent_1688

python -m scripts.download_model \
        --model allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219 \
        --serialization_dir ${out_dir}/pretrained_models/dsp_roberta_base_dapt_cs_tapt_sciie_3219

python -m scripts.download_model \
        --model allenai/dsp_roberta_base_tapt_sciie_3219 \
        --serialization_dir ${out_dir}/pretrained_models/dsp_roberta_base_tapt_sciie_3219

