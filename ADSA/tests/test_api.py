import requests
import os

url = "http://127.0.0.1:8000/api/upload"
file_path = "sample_data/churn.csv"

if not os.path.exists(file_path):
    print("Sample data not found")
else:
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"user_goal": "predict churn"}
        try:
            response = requests.post(url, files=files, data=data)
            print("Status Code:", response.status_code)
            print("Response:", response.text)
        except Exception as e:
            print("Error:", e)
