from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN")

api = HfApi()
user = api.whoami(token=token)

print("You are logged in as:", user["name"])
