import requests
import json

url = "http://localhost:8000/resume_ranking/rank"
payload = {
    "job_description": {
        "title": "Artificial Intelligence / Machine Learning Intern",
        "description": "We are looking for a motivated AI/ML Intern...",
        "required_skills": ["Python", "Machine Learning"]
    },
    "candidates": [
        {
            "name": "John Doe",
            "experience": "2 years of Python",
            "education": "B.S. Computer Science",
            "skills": ["Python", "Machine Learning"]
        }
    ]
}

print("Sending request to:", url)
response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
try:
    print(json.dumps(response.json(), indent=2))
except:
    print(response.text)
