import fitz
import requests
import json

print("1. Creating dummy resume PDF...")
doc = fitz.open()
page = doc.new_page()
text = """Alice Smith
Email: alice.smith@example.com
Phone: 555-0199
Location: New York, NY

Summary
Experienced software engineer with a background in Python, machine learning, and web development.

Experience
Senior Python Developer at DataWorks (2018 - Present)
- Developed highly scalable REST APIs using Python and FastAPI.
- Integrated AI models using PyTorch and HuggingFace.
- Improved database query performance by 40%.

Education
B.S. in Computer Science - State University

Skills
Technical: Python, FastAPI, PyTorch, Docker, SQL
"""
page.insert_text(fitz.Point(50, 50), text, fontsize=11)
doc.save("test_resume.pdf")
print("[SUCCESS] Saved test_resume.pdf")

print("\n2. Calling /unified/pipeline API...")
url = "http://localhost:8000/unified/pipeline"
data = {"job_description": "We are looking for a Senior AI Software Engineer with strong Python and FastAPI skills. Experience with PyTorch or machine learning is a huge plus."}

with open("test_resume.pdf", "rb") as f:
    files = [("files", ("test_resume.pdf", f, "application/pdf"))]
    response = requests.post(url, data=data, files=files)

print(f"\nStatus Code: {response.status_code}")
if response.status_code == 200:
    print("[SUCCESS] Response:")
    print(json.dumps(response.json(), indent=2))
else:
    print("[FAILED] Response:")
    print(response.text)
