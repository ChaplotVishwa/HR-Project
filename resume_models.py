from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any

class Experience(BaseModel):
    company: str
    position: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    responsibilities: List[str] = []

    @validator('start_date', 'end_date', pre=True)
    def normalize_dates(cls, v):
        if v and v.lower() in ['present', 'current', 'now']:
            return 'Present'
        return v

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    location: Optional[str] = None

class PersonalInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

class Skills(BaseModel):
    technical: List[str] = []
    soft: List[str] = []
    languages: List[str] = []
    tools: List[str] = []

class ResumeData(BaseModel):
    personal_info: PersonalInfo
    summary: Optional[str] = None
    experience: List[Experience] = []
    education: List[Education] = []
    skills: Skills = Skills()
    certifications: List[str] = []
    projects: List[Dict[str, Any]] = []
    achievements: List[str] = []
