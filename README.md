# ArecaCare AI Backend

FastAPI backend for the ArecaCare React frontend.

## 1. Create environment

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Start API

```powershell
uvicorn main:app --reload --port 8000
```

Check:

`http://127.0.0.1:8000/ping`

## 3. Prediction endpoint

`POST http://127.0.0.1:8000/api/predict`

Form field: `file`

The API uses the supplied 9-class Arecanut CNN model.

Classes:
- Healthy Leaf
- Healthy Nut
- Healthy Trunk
- Mahali / Koleroga
- Stem Bleeding
- Bud Borer
- Healthy Foot
- Stem Cracking
- Yellow Leaf Disease

## Important

The model is an existing project model. Treat its prediction as preliminary and verify agricultural treatment decisions with an appropriate agriculture professional.
