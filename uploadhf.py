from huggingface_hub import create_repo, upload_folder

repo_id = "haroldevvv/my-mbart50-translation-model"
local_dir = r"C:\Users\admin\Desktop\Prototype_MT\mBART50_augmented_direct\runs\rin_en_3.5k_15epochs_mbart50_run-20251006-115936\final_model"

create_repo(repo_id, repo_type="model", exist_ok=True)

upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned mBART50 model"
)

print(f" Model uploaded successfully to https://huggingface.co/{repo_id}")