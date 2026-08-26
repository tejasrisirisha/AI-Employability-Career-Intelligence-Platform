"""
generate_dataset.py
--------------------
Generates a realistic, logic-grounded synthetic dataset for the
AI Student Employability & Placement Prediction System.

Why synthetic-but-grounded?
Public campus-placement datasets (e.g. Kaggle "Campus Recruitment") only have
~215 rows and ~10 columns, which is too thin for a robust, explainable,
multi-feature system (skills, aptitude, certifications, soft-skills, etc.).
Instead we generate a large (6000-row) dataset whose PLACEMENT LABEL is
produced by a transparent, weighted scoring function built from real
placement-research factors (CGPA, internships, projects, aptitude,
communication, technical skill, backlogs, certifications, etc.), then we add
realistic noise. This gives the ML model genuine, learnable signal (so
predictions are meaningful, not random) while still being fully reproducible
and shareable as a CSV.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 6000
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

branches = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]
branch_weight = {"CSE": 1.00, "IT": 0.98, "ECE": 0.88, "EEE": 0.82, "MECH": 0.75, "CIVIL": 0.72}

genders = ["Male", "Female"]

rows = []
for i in range(N):
    branch = np.random.choice(branches, p=[0.30, 0.20, 0.20, 0.12, 0.10, 0.08])
    gender = np.random.choice(genders, p=[0.58, 0.42])

    cgpa = np.clip(np.random.normal(7.2, 1.1), 4.0, 10.0)
    ssc_marks = np.clip(np.random.normal(78, 10), 40, 100)
    hsc_marks = np.clip(np.random.normal(75, 11), 40, 100)

    internships = np.random.choice([0, 1, 2, 3, 4], p=[0.30, 0.32, 0.22, 0.11, 0.05])
    projects = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.08, 0.15, 0.22, 0.22, 0.16, 0.11, 0.06])
    certifications = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.20, 0.25, 0.22, 0.16, 0.10, 0.07])
    backlogs = np.random.choice([0, 1, 2, 3, 4], p=[0.55, 0.22, 0.13, 0.07, 0.03])

    aptitude_score = np.clip(np.random.normal(62, 18), 0, 100)
    technical_skill = np.clip(np.random.normal(6.0, 1.8), 0, 10)
    coding_skill = np.clip(technical_skill + np.random.normal(0, 1.2), 0, 10)
    communication_skill = np.clip(np.random.normal(6.2, 1.7), 0, 10)
    soft_skill = np.clip((communication_skill + np.random.normal(6.0, 1.6)) / 2, 0, 10)
    extracurricular = np.clip(np.random.normal(5.5, 2.2), 0, 10)
    leadership = np.clip(np.random.normal(5.0, 2.2), 0, 10)

    placement_training = np.random.choice(["Yes", "No"], p=[0.55, 0.45])
    workshops = np.random.choice([0, 1, 2, 3, 4], p=[0.30, 0.30, 0.20, 0.12, 0.08])
    live_kt_score = np.clip(np.random.normal(5.5, 1.8), 0, 10)  # knowledge test / mock interview
    linkedin_github_activity = np.clip(np.random.normal(5.0, 2.3), 0, 10)

    # ---- Weighted, research-informed employability score (0-100) ----
    score = (
        cgpa / 10 * 20 +
        np.clip(internships, 0, 3) / 3 * 12 +
        np.clip(projects, 0, 5) / 5 * 10 +
        np.clip(certifications, 0, 4) / 4 * 6 +
        aptitude_score / 100 * 14 +
        technical_skill / 10 * 12 +
        coding_skill / 10 * 8 +
        communication_skill / 10 * 8 +
        soft_skill / 10 * 4 +
        (1 if placement_training == "Yes" else 0) * 3 +
        extracurricular / 10 * 2 +
        leadership / 10 * 1
    )
    score = score * branch_weight[branch]
    score = score - backlogs * 4.2
    score = score + np.random.normal(0, 3.5)  # real-world noise (kept modest so the signal stays learnable)
    score = np.clip(score, 0, 100)

    prob_placed = 1 / (1 + np.exp(-(score - 55) / 5.5))  # sigmoid centered ~55
    placed = np.random.binomial(1, prob_placed)

    # Placement package (LPA) only meaningful if placed
    base_package = 3.0 + (score / 100) * 9.0 + (1 if branch in ["CSE", "IT"] else 0) * 1.2
    package = np.round(np.clip(base_package + np.random.normal(0, 0.6), 2.4, 24.0), 2) if placed else 0.0

    rows.append({
        "Student_ID": f"STU{i+1:05d}",
        "Gender": gender,
        "Branch": branch,
        "CGPA": round(cgpa, 2),
        "SSC_Marks": round(ssc_marks, 2),
        "HSC_Marks": round(hsc_marks, 2),
        "Internships": int(internships),
        "Projects": int(projects),
        "Certifications": int(certifications),
        "Backlogs": int(backlogs),
        "Aptitude_Score": round(aptitude_score, 2),
        "Technical_Skill": round(technical_skill, 2),
        "Coding_Skill": round(coding_skill, 2),
        "Communication_Skill": round(communication_skill, 2),
        "Soft_Skill": round(soft_skill, 2),
        "Extracurricular_Score": round(extracurricular, 2),
        "Leadership_Score": round(leadership, 2),
        "Placement_Training": placement_training,
        "Workshops_Attended": int(workshops),
        "Mock_Interview_Score": round(live_kt_score, 2),
        "LinkedIn_GitHub_Activity": round(linkedin_github_activity, 2),
        "Employability_Score": round(score, 2),
        "Placed": int(placed),
        "Package_LPA": package,
    })

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "student_employability_dataset.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows -> {out_path}")
print(df["Placed"].value_counts(normalize=True))
print(df.describe(include="all").T.head(10))
