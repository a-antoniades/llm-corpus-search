CUDA_VISIBLE_DEVICES="" python wimbd_preprocess_translation.py \
    --base_dir "./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue"

CUDA_VISIBLE_DEVICES="" python wimbd_preprocess_translation.py \
    --base_dir "./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue"

