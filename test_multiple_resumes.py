import fitz
import requests
import json
import os

print("1. Creating dummy resume PDFs...")

# Resume 1: Perfect Match (Senior Python & AI Engineer)
doc1 = fitz.open()
page1 = doc1.new_page()
text1 = """Alice Perfect
Email: alice.perfect@example.com
Phone: 555-0001
Location: San Francisco, CA

Summary
Expert Senior Software Engineer with 8 years of experience building scalable AI applications using Python, FastAPI, and PyTorch.

Experience
Lead AI Engineer at TechNova (2018 - Present)
- Architected and deployed machine learning models using PyTorch and HuggingFace.
- Built high-performance microservices using Python and FastAPI, handling 1M+ daily requests.
- Mentored junior engineers and led Agile sprints.

Education
M.S. in Computer Science - Stanford University

Skills
Technical: Python, FastAPI, PyTorch, Machine Learning, Docker, Kubernetes, AWS
Soft: Leadership, Communication, Agile
"""
page1.insert_text(fitz.Point(50, 50), text1, fontsize=11)
doc1.save("resume_alice_perfect.pdf")


# Resume 2: Partial Match (Java Backend Developer, learning AI)
doc2 = fitz.open()
page2 = doc2.new_page()
text2 = """Bob Partial
Email: bob.partial@example.com
Phone: 555-0002
Location: Austin, TX

Summary
Backend Developer with 5 years of experience in Java and Spring Boot. Recently completed a certification in Machine Learning and Python.

Experience
Software Engineer at EnterpriseCorp (2020 - Present)
- Developed REST APIs using Java and Spring Boot.
- Managed PostgreSQL databases and optimized queries.
- Created an internal tool using simple Python scripts.

Education
B.S. in Software Engineering - Texas A&M

Skills
Technical: Java, Spring Boot, SQL, Git, basic Python
Soft: Teamwork, Problem Solving
"""
page2.insert_text(fitz.Point(50, 50), text2, fontsize=11)
doc2.save("resume_bob_partial.pdf")


# Resume 3: Poor Match (Frontend React Developer)
doc3 = fitz.open()
page3 = doc3.new_page()
text3 = """Charlie Poor
Email: charlie.poor@example.com
Phone: 555-0003
Location: Seattle, WA

Summary
Creative Frontend Developer specializing in React and CSS animations. Passionate about building beautiful UIs.

Experience
Frontend Developer at Webify (2021 - Present)
- Built responsive web pages using React and Tailwind CSS.
- Collaborated with UX designers to implement wireframes.
- Optimized frontend performance reducing bundle size by 20%.

Education
B.A. in Graphic Design - Seattle University

Skills
Technical: JavaScript, React, HTML, CSS, Figma
Soft: Creativity, Design Thinking
"""
page3.insert_text(fitz.Point(50, 50), text3, fontsize=11)
doc3.save("resume_charlie_poor.pdf")

print("[SUCCESS] Created 3 test resumes.")

print("\n2. Calling /unified/pipeline API...")
url = "http://localhost:8000/unified/pipeline"

# Job Description clearly targets Python, FastAPI, PyTorch, AI
job_desc = """We are looking for a Senior AI Software Engineer.
The ideal candidate must have strong experience in Python and building REST APIs with FastAPI.
Experience with machine learning frameworks like PyTorch and building AI applications is essential.
Candidates should have 5+ years of experience and leadership skills."""

data = {"job_description": job_desc}

# Open all 3 files
files = [
    ("files", ("resume_alice_perfect.pdf", open("resume_alice_perfect.pdf", "rb"), "application/pdf")),
    ("files", ("resume_bob_partial.pdf", open("resume_bob_partial.pdf", "rb"), "application/pdf")),
    ("files", ("resume_charlie_poor.pdf", open("resume_charlie_poor.pdf", "rb"), "application/pdf"))
]

try:
    response = requests.post(url, data=data, files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    if response.status_code == 200:
        print("[SUCCESS] Response:")
        res_json = response.json()
        print(f"Algorithm Used: {res_json.get('metadata', {}).get('algorithm')}")
        print("-" * 50)
        
        for cand in res_json.get("results", []):
            print(f"Rank: #{cand['rank']}")
            print(f"Name: {cand['name']}")
            print(f"Score: {cand['total_score']}%")
            print(f"Matched Skills: {', '.join(cand['skills'][:5])}...")
            print("-" * 50)
            
    else:
        print("[FAILED] Response:")
        print(response.text)
finally:
    # Close files
    for _, (_, f, _) in files:
        f.close()
    
    # Cleanup dummy files
    for f in ["resume_alice_perfect.pdf", "resume_bob_partial.pdf", "resume_charlie_poor.pdf"]:
        if os.path.exists(f):
            os.remove(f)
