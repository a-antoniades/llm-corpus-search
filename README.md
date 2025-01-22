# LLM-Incidental-Supervision
## Incidental Supervision

### Environment Setup

Install the wimbd repo, and checkout the commit that I used.
```bash
git clone git@github.com:allenai/wimbd.git
cd wimbd
git checkout 0b787c31f856a512f6289657eef10f852b2f936e
cd ..
```

Once you pull the repo, I added a `requirements.txt` file which you can use to install the dependencies.

### Searching for N-grams in PILE

Here is an example of how to search for n-grams from some subtasks from MMLU in PILE.

```bash
CUDA_VISIBLE_DEVICES=1 python wimbd_search.py \
                        --type infini \
                        --corpus pile \
                        --n_grams 2 \
                        --dataset mmlu \
                        --replace_keywords false \
                        --filter_stopwords true \
                        --only_alpha false \
                        --name exp5 \
                        --align_pairs true \
                        --tasks marketing \
                                management \
                                high_school_world_history \
                                high_school_european_history \
                                miscellaneous \
                        --method common
```