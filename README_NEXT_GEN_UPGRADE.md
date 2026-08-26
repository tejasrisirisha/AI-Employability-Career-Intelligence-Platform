# Next-Gen AI Employability & Career Intelligence Upgrade

This upgrade keeps the existing application flow and legacy features, while adding a compact single **🧭 Navigate** control in the sidebar.

## Existing features preserved
- Student Profile
- Overview & Placement Prediction
- Explainable AI
- Skill Gap Analysis
- Job Role Matching
- Resume Analysis
- Personalized Improvement Plan
- What-If Simulator
- Progress Tracking

## New highlighted features
- AI Career Mentor
- Placement Cell Dashboard
- Company-Specific Readiness
- Career DNA & Strengths Profile
- Risk Diagnosis — Why might I not be placed?
- Counterfactual Optimizer
- Skill Trend & Market Radar
- Interview Lab / Mock Interviews
- Placement Day Simulator
- Intervention Impact Simulator
- Autonomous 90-Day Career Planner
- Professional Career Report + Readiness Certificate
- Fairness & Bias Audit
- Gamification & Career Badges
- SHAP-style Waterfall Visualization inside Explainable AI

## Important
The app still expects your existing project assets:
- `job_roles.py`
- `models/placement_classifier.pkl`
- `models/package_regressor.pkl`
- `models/scaler.pkl`
- `models/label_encoders.pkl`
- `models/feature_columns.pkl`
- `models/metrics.json`
- `data/student_employability_dataset.csv`
- `data/progress_log.csv` (created automatically if missing)

## Run
```bash
pip install -r requirements_next_gen.txt
streamlit run app_next_gen.py
```

The new mentor, simulations, readiness benchmarks and market radar are designed to work without an external API. Company benchmarks and market signals are clearly presented as planning/demo benchmarks rather than official hiring requirements or live labor-market feeds.
