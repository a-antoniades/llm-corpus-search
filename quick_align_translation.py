df1 = lang_dfs['pythia-160m']
df2 = model_df_languages_aligned

# Merging the dataframes on 'pair' and 'task' columns
merged_df = pd.merge(df1, df2[['pair', 'task', 'alignment_score']], on=['pair', 'task'], how='left')

# Displaying the merged dataframe
print(merged_df)

lang_df_new = {}

for model in lang_dfs.keys():
    lang_df_new[model] = pd.merge(lang_dfs[model], model_df_languages_aligned[['pair', 'task', 'alignment_score']], on=['pair', 'task'], how='left')


# join aligment score column to lang_dfs
for model in lang_dfs.keys():
    model_df = lang_dfs[model]
    model_df_languages = {task: model_df[model_df['task'] == task] for task in model_df['task'].unique()}
    for language_pair in model_df_languages.keys():
        alignment_score = model_df_languages[language_pair]['alignment_score']
        lang_dfs[model]

# save
with open(LANG_DF_PTH, 'wb') as f:
    pickle.dump(lang_df_new, f)

