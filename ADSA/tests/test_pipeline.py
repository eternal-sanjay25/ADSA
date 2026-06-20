import requests
import json
import sys
import os

url_upload = "http://127.0.0.1:7860/api/upload"
file_path = "sample_data/churn.csv"

with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"user_goal": "predict churn"}
    res = requests.post(url_upload, files=files, data=data)
    session_id = res.json()["session_id"]

url_run = f"http://127.0.0.1:7860/api/run/{session_id}"

print(f"Starting pipeline for session: {session_id}")
response = requests.get(url_run, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

