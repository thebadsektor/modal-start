from huggingface_hub import login, upload_folder
import os

# Login with your token
hf_token = input("Enter your HuggingFace token: ")
login(token=hf_token)

# Your HuggingFace username
username = input("Enter your HuggingFace username: ")
repo_name = f"{username}/mistral-7b-viral-posts"

print(f"Pushing to: https://huggingface.co/{repo_name}")

# Upload model
upload_folder(
    folder_path="./my_model",
    repo_id=repo_name,
    repo_type="model",
)

print(f"✅ Model uploaded to: https://huggingface.co/{repo_name}")
