import os
import glob
import shutil

file_names = ['doc_results.json', 'results.json']

src_pth = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_6_logits_max_3"
dest_pth = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_6_logits_max_4"

doc_res_files = glob.glob(os.path.join(src_pth, "**/doc_results.json"), recursive=True)
res_files = glob.glob(os.path.join(src_pth, "**/results.json"), recursive=True)

print("doc_res_files:", len(doc_res_files))
print("res_files:", len(res_files))

# # Copy and overwrite doc_results.json files
# for file in doc_res_files:
#     dest_file = file.replace(src_pth, dest_pth)
#     os.makedirs(os.path.dirname(dest_file), exist_ok=True)
#     shutil.copy2(file, dest_file)  # This will copy and overwrite the file if it exists


# Copy and overwrite results.json files
for file in res_files:
    dest_file = file.replace(src_pth, dest_pth)
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    shutil.copy2(file, dest_file)  # This will copy and overwrite the file if it exists
