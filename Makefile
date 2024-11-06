# Define the language pairs
# LANG_PAIRS = fr_en de_en ja_en ro_en ru_en zh_en cs_en pl_en
# # first exp mix
# LANG_PAIRS = zh_en fr_en de_en ja_en ro_en ru_en zh_en cs_en pl_en

## wmt09
# LANG_PAIRS = cs_en de_en fr_en es_en it_en lt_en hu_en zh_en ja_en ru_en ro_en pl_en
# LANG_PAIRS = cs_en de_en fr_en es_en it_en hu_en
# LANG_PAIRS = en_cs en_de en_fr en_es en_it en_hu

# europarl
# LANG_PAIRS = cs_en en_fr de_en en_hu en_es en_it
LANG_PAIRS = en_fr

N_GRAMS = 1 2 3 4 5
CORPUS = pile
DATASET = wmt09_gens
N_SAMPLES = 100

# models pythia-410m pythia-12b pythia-2.8b 

# Define the all rule
all:
	for ngram in $(N_GRAMS); do \
		GPU=$$ngram; \
		for pair in $(LANG_PAIRS); do \
			SESSION_NAME=$$(echo $$pair-$$ngram-$(CORPUS)-$(DATASET)-$(N_SAMPLES)-2 | tr -d ' '); \
			LANG1=$$(echo $$pair | cut -d'_' -f1); \
			LANG2=$$(echo $$pair | cut -d'_' -f2); \
			echo "Killing any existing session named $$SESSION_NAME"; \
			tmux kill-session -t $$SESSION_NAME || true; \
			echo "Running script with language pair $$LANG1 $$LANG2 on GPU $$GPU"; \
			tmux new-session -d -s $$SESSION_NAME bash -c "\
			OMP_NUM_THREADS=1 \
			MKL_NUM_THREADS=1 \
			CUDA_VISIBLE_DEVICES=$$GPU \
			python wimbd_search.py \
				--type infini \
				--corpus $(CORPUS) \
				--n_grams $$ngram \
				--dataset $(DATASET) \
				--tasks pythia-410m pythia-12b pythia-2.8b pythia-160m  \
				--language_pair '$$LANG1' '$$LANG2' \
				--filter_keywords false \
				--replace_keywords false \
				--filter_stopwords false \
				--align_pairs false \
				--method common \
				--only_alpha true \
				--get_docs false \
				--n_samples $(N_SAMPLES) \
				--name exp4__/"; \
			GPU=$$(( (GPU+1) % 8 )); \
		done \
	done
.PHONY: all


# # mmlu
# DATASET = triviaqa # sciq # mmlu
# CORPUS = dolma # pile
# N_GRAMS = 3
# NAME = exp_3/validation-set
# N_SAMPLES = 20000

# # Define the all rule`
# all:
# 	GPU=""; \
# 	echo "Running script with ngram $(N_GRAMS) on GPU $$GPU"; \
# 	SESSION_NAME=$$(echo "wimbd-$(DATASET)-$(CORPUS)-$(N_GRAMS)-$(N_SAMPLES)" | tr -d ' '); \
# 	echo "Killing any existing session named $$SESSION_NAME"; \
# 	tmux kill-session -t $$SESSION_NAME || true; \
# 	tmux new-session -d -s $$SESSION_NAME 'bash -c "\
# 	CUDA_VISIBLE_DEVICES="" \
# 	python wimbd_search.py \
# 	--type infini \
# 	--corpus $(CORPUS) \
# 	--n_grams $(N_GRAMS) \
# 	--dataset $(DATASET) \
# 	--filter_punc true \
# 	--filter_stopwords true \
# 	--filter_keywords false \
# 	--replace_keywords false \
# 	--corpus $(CORPUS) \
# 	--only_alpha false \
# 	--method common \
# 	--name \"$(NAME)\"' & \
# 	GPU=$$(( (GPU+2) % 8 )); \

# .PHONY: all


# --delimeter \" = \" \
# bigbench
# N_GRAMS = 2 3 4 5 6
# N_GRAMS = 5

# # Define the all rule
# all:
# 	for ngram in $(N_GRAMS); do \
# 		SESSION_NAME=$$(echo "bigbench-wimbd-$$ngram" | tr -d ' '); \
# 		echo "Killing any existing session named $$SESSION_NAME"; \
# 		tmux kill-session -t $$SESSION_NAME || true; \
# 		tmux new-session -d -s $$SESSION_NAME 'bash -c "CUDA_VISIBLE_DEVICES=0 python wimbd_search.py --n_grams '$$ngram' --dataset bigbench --n_samples 200; bash"' & \
# 		GPU=$$(( (GPU+1) % 8 )); \
# 	done

# .PHONY: all