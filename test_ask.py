import requests

url = "http://127.0.0.1:5000/ask"
data = {"query": "What are the signs of a malignant lung nodule?"}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
