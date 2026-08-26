"""
job_roles.py
------------
Reference profiles used for Job Role Matching and Skill Gap Analysis.
Each role defines ideal target levels (0-10 scale, CGPA on 0-10, Aptitude 0-100)
built from common industry hiring expectations. These are intentionally
transparent/editable so the app's logic stays explainable.
"""

JOB_ROLES = {
    "Software Developer": {
        "icon": "💻",
        "weights": {
            "CGPA": 6.5, "Technical_Skill": 7.5, "Coding_Skill": 8.0,
            "Projects": 4, "Aptitude_Score": 65, "Communication_Skill": 5.5,
            "Certifications": 2,
        },
        "core_skills": ["Data Structures & Algorithms", "OOP", "Git/GitHub", "SQL", "Problem Solving"],
        "description": "Builds and maintains software applications across the stack.",
    },
    "Data Scientist / ML Engineer": {
        "icon": "📊",
        "weights": {
            "CGPA": 7.5, "Technical_Skill": 8.5, "Coding_Skill": 7.5,
            "Projects": 5, "Aptitude_Score": 75, "Certifications": 4,
            "Communication_Skill": 6.0,
        },
        "core_skills": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
        "description": "Builds predictive models and derives insight from data.",
    },
    "Web Developer": {
        "icon": "🌐",
        "weights": {
            "CGPA": 6.0, "Technical_Skill": 6.5, "Coding_Skill": 7.0,
            "Projects": 5, "Aptitude_Score": 55, "Communication_Skill": 5.5,
        },
        "core_skills": ["HTML/CSS", "JavaScript", "React/Frontend Framework", "REST APIs", "Git"],
        "description": "Designs and develops responsive web applications.",
    },
    "Business/Data Analyst": {
        "icon": "📈",
        "weights": {
            "CGPA": 6.5, "Aptitude_Score": 72, "Communication_Skill": 8.0,
            "Soft_Skill": 7.0, "Technical_Skill": 5.5, "Certifications": 2,
        },
        "core_skills": ["Excel", "SQL", "Power BI/Tableau", "Communication", "Business Acumen"],
        "description": "Turns business data into actionable decisions and reports.",
    },
    "System/Network Engineer": {
        "icon": "🖥️",
        "weights": {
            "CGPA": 6.0, "Technical_Skill": 6.5, "Backlogs_Inverse": 9,
            "Aptitude_Score": 58, "Certifications": 3,
        },
        "core_skills": ["Networking Basics", "Linux", "Cloud Fundamentals", "Troubleshooting", "Security Basics"],
        "description": "Maintains and secures IT infrastructure and networks.",
    },
    "Core/Non-IT Engineer": {
        "icon": "⚙️",
        "weights": {
            "CGPA": 6.5, "Technical_Skill": 6.0, "Projects": 3,
            "Aptitude_Score": 55, "Communication_Skill": 5.0,
        },
        "core_skills": ["Domain Fundamentals", "CAD/Design Tools", "Industry Standards", "Teamwork", "Problem Solving"],
        "description": "Applies core engineering knowledge in domain-specific industries.",
    },
}

# Resume keyword bank used for automatic resume skill extraction
RESUME_SKILL_KEYWORDS = {
    "Programming": ["python", "java", "c++", "c programming", "javascript", "typescript", "r programming", "sql", "golang"],
    "Web": ["html", "css", "react", "angular", "node.js", "django", "flask", "rest api", "bootstrap"],
    "Data/ML": ["machine learning", "deep learning", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
                "data analysis", "power bi", "tableau", "nlp", "computer vision"],
    "Cloud/DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "git", "github", "linux"],
    "Soft Skills": ["leadership", "teamwork", "communication", "presentation", "problem solving", "public speaking"],
    "Certifications": ["certified", "certification", "nptel", "coursera", "udemy", "hackerrank"],
    "Database": ["mysql", "postgresql", "mongodb", "oracle", "database management"],
}
