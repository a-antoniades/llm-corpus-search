# Define the language pairs
# LANG_PAIRS = fr_en de_en ja_en ro_en ru_en zh_en cs_en pl_en
# # first exp mix
# LANG_PAIRS = zh_en fr_en de_en ja_en ro_en ru_en zh_en cs_en pl_en

## wmt09
# LANG_PAIRS = cs_en de_en fr_en es_en it_en lt_en hu_en zh_en ja_en ru_en ro_en pl_en
# LANG_PAIRS = cs_en de_en fr_en es_en it_en hu_en
# LANG_PAIRS = en_cs en_de en_fr en_es en_it en_hu

# # europarl
# LANG_PAIRS = cs_en en_fr de_en en_hu en_es en_it

# N_GRAMS = 5


# # Define the all rule
# all:
# 	GPU=6; \
# 	for pair in $(LANG_PAIRS); do \
# 		SESSION_NAME=$$(echo $$pair-$(N_GRAMS) | tr -d ' '); \
# 		LANG1=$$(echo $$pair | cut -d'_' -f1); \
# 		LANG2=$$(echo $$pair | cut -d'_' -f2); \
# 		echo "Killing any existing session named $$SESSION_NAME"; \
# 		tmux kill-session -t $$SESSION_NAME || true; \
# 		echo "Running script with language pair $$LANG1 $$LANG2 on GPU $$GPU"; \
# 		tmux new-session -d -s $$SESSION_NAME 'bash -c "\
# 		CUDA_VISIBLE_DEVICES=6 \
# 		python wimbd_search.py \
# 			--type infini \
# 			--corpus pile \
# 			--n_grams $(N_GRAMS) \
# 			--dataset europarl \
# 			--n_samples 20000 \
# 			--language_pair '$$LANG1' '$$LANG2' \
# 			--corpus pile \
# 			--filter_stopwords false \
# 			--filter_keywords false \
# 			--replace_keywords false \
# 			--only_alpha true \
# 			--get_docs false \
# 			--name exp4/"' & \
# 		GPU=$$(( (GPU+1) % 8 )); \
# 	done
# .PHONY: all

# mmlu
DATASET = sciq # mmlu # trivia_qa
CORPUS = dolma
N_GRAMS = 5
NAME = exp4_infini/
N_SAMPLES = None

# Define the all rule
all:
	GPU=7; \
	echo "Running script with ngram $(N_GRAMS) on GPU $$GPU"; \
	SESSION_NAME=$$(echo "wimbd-$(DATASET)-$(CORPUS)-$(N_GRAMS)-$(N_SAMPLES)" | tr -d ' '); \
	echo "Killing any existing session named $$SESSION_NAME"; \
	tmux kill-session -t $$SESSION_NAME || true; \
	tmux new-session -d -s $$SESSION_NAME 'bash -c "\
	CUDA_VISIBLE_DEVICES="" \
	python wimbd_search.py \
	--type infini \
	--corpus $(CORPUS) \
	--n_grams $(N_GRAMS) \
	--dataset $(DATASET) \
	--filter_stopwords true \
	--filter_keywords false \
	--replace_keywords false \
	--corpus $(CORPUS) \
	--only_alpha false \
	--name \"$(NAME)\"; exec bash"' & \
	GPU=$$(( (GPU+2) % 8 )); \

.PHONY: all

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