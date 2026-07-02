# AI Fraud Detection System

A production-ready AI Fraud Detection System boilerplate with a FastAPI backend, Streamlit frontend, SQLite database, and Scikit-learn integration.

## Project Structure

```
Fraud_Detection/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application entrypoint
│   │   ├── database.py      # SQLite connection and session details
│   │   ├── models.py        # SQLAlchemy database model definitions
│   │   ├── schemas.py       # Pydantic schemas for verification/exchange
│   │   ├── crud.py          # CRUD operations logic
│   │   └── ml/
│   │       ├── __init__.py
│   │       └── model.py     # Scikit-learn random forest integration wrapper
├── frontend/
│   └── app.py               # Streamlit application UI
├── requirements.txt         # Common requirements list
└── README.md
```


# 1. Navigate to the project directory (if not already there)
cd /Users/NIRAJKUMAR/Desktop/Fraud_Detection

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt


## Running the Application

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start FastAPI Backend**:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

3. **Start Streamlit Frontend**:
   ```bash
   streamlit run frontend/app.py --server.port 8501
   ```



# In your first terminal (with venv activated):
# uvicorn backend.app.main:app --reload --port 8000

# In your second terminal (activate venv and run):
# source venv/bin/activate
# streamlit run frontend/app.py --server.port 8501
