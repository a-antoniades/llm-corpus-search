# LLM-Incidental-Supervision
## Incidental Supervision

### Environment Setup

Once you pull the repo, I added a `requirements.txt` file, try to create your environment with that, although it may not work. In that case, please comment out the culprits from the file, and install the manually.

### Plotting Translation results

Use `plot_translation.py` and set your `LANG_DF_PTH`. The file was made to be readily compatible with jupyter notebook execution.


### Viewing Ngram Search Dataframe (Xinyi)

See `wimbd_translation_datasets.ipynb` for an example of how to load the dataframes and what you should see. You can see there is a column "docs" which shows from where in the pretraining corpus (PILE) the data was obtained. Let me know if you have any questions!