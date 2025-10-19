# %%

from huggingface_hub import snapshot_download
import os

repo_id = "adamkarvonen/checkpoints_all_single_and_multi_pretrain_only_Qwen3-8B"

folder = repo_id.split("/")[-1]
os.makedirs(folder, exist_ok=True)
folder = os.path.join(folder, "final")
os.makedirs(folder, exist_ok=True)

# snapshot_download(repo_id=repo_id, local_dir=folder, allow_patterns="step_5000*")


snapshot_download(repo_id=repo_id, local_dir=folder)

# %%
