import requests

url = "http://127.0.0.1:5000/predict"
data = {"DNA_sequence": "ATGCGTACGTTAGC"}

response = requests.post(url, json=data)
print(response.json())
