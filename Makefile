# Define the language pairs
# LANG_PAIRS = fr_en de_en ja_en ro_en ru_en zh_en cs_en pl_en
# # first exp mix
# LANG_PAIRS = zh_en fr_en de_en ja_en ro_en ru_en zh_en cs_en pl_en

# # wmt09
# LANG_PAIRS = cs_en de_en fr_en es_en it_en lt_en hu_en zh_en ja_en ru_en ro_en pl_en
# LANG_PAIRS = cs_en de_en fr_en es_en it_en hu_en
LANG_PAIRS = en_cs en_de en_fr en_es en_it en_hu
N_GRAMS = 3


# # Define the all rule
all:
	GPU=2; \
	for pair in $(LANG_PAIRS); do \
		SESSION_NAME=$$(echo $$pair-$(N_GRAMS) | tr -d ' '); \
		LANG1=$$(echo $$pair | cut -d'_' -f1); \
		LANG2=$$(echo $$pair | cut -d'_' -f2); \
		echo "Killing any existing session named $$SESSION_NAME"; \
		tmux kill-session -t $$SESSION_NAME || true; \
		echo "Running script with language pair $$LANG1 $$LANG2 on GPU $$GPU"; \
		tmux new-session -d -s $$SESSION_NAME 'bash -c "\
		CUDA_VISIBLE_DEVICES='$$GPU' \
		python wimbd_search.py \
			--n_grams $(N_GRAMS) \
			--dataset wmt \
			--language_pair '$$LANG1' '$$LANG2' \
			--corpus pile \
			--filter_stopwords true \
			--filter_keywords false \
			--replace_keywords false \
			--only_alpha true \
			--get_docs false \
			--name exp4/; exec bash"' & \
		GPU=$$(( (GPU+1) % 8 )); \
	done

.PHONY: all

# # mmlu
# # GPU=$$(( (GPU+1) % 8 )); \
# N_GRAMS = 1 2 4 5
# DATASET = trivia_qa # mmlu
# CORPUS = pile
# N_GRAMS = 5
# NAME = exp4/

# # Define the all rule
# all:
# 	GPU=2; \
# 	for ngram in $(N_GRAMS); do \
# 		echo "Running script with ngram $$ngram on GPU $$GPU"; \
# 		SESSION_NAME=$$(echo "wimbd-$(DATASET)-$(CORPUS)-$$ngram-train" | tr -d ' '); \
# 		echo "Killing any existing session named $$SESSION_NAME"; \
# 		tmux kill-session -t $$SESSION_NAME || true; \
# 		tmux new-session -d -s $$SESSION_NAME 'bash -c "\
# 		CUDA_VISIBLE_DEVICES=1 \
# 		python wimbd_search.py \
# 		--corpus $(CORPUS) \
# 		--n_grams 5 \
# 		--dataset $(DATASET) \
# 		--filter_keywords false \
# 		--replace_keywords false \
# 		--method common \
# 		--only_alpha false \
# 		--n_samples 10000 \
# 		--name \"$(NAME)\"; exec bash"' & \
# 		GPU=$$(( (GPU+2) % 8 )); \
# 	done

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