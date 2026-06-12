# ArthriCare — Arthritis Health Management App

## Quick Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your Supabase URL and Key
```

### 3. Run SQL schema
Paste the schema SQL into Supabase SQL Editor and run it.

### 4. Run app
```bash
python app.py
```
Visit: http://127.0.0.1:5000

## Project Structure
```
arthricare/
├── app.py              ← Flask entry point
├── db.py               ← Supabase connection
├── requirements.txt
├── .env                ← Your secrets (never commit)
├── models/
│   └── user.py         ← Flask-Login user model
├── routes/
│   ├── auth.py         ← Login, Register, Logout
│   ├── patient.py      ← All patient features
│   ├── doctor.py       ← Doctor/Researcher dashboard
│   └── ml_routes.py    ← AI prediction endpoints
├── ml/
│   ├── hybrid_model.py ← SVM + Random Forest classifier
│   └── image_model.py  ← X-ray analysis + Grad-CAM
└── templates/
    ├── base.html
    ├── landing.html
    ├── login.html / register.html
    ├── patient_dashboard.html
    ├── pain_tracker.html
    ├── heart_rate.html
    ├── diet_plan.html
    ├── exercise_plan.html
    ├── upload_report.html
    ├── ml_prediction.html
    ├── monitoring.html
    └── doctor_dashboard.html
```
