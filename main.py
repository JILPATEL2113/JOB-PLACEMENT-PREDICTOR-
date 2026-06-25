from fastapi import FastAPI
from fastapi.responses import HTMLResponse 
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Job Placement Prediction API")


try:
    with open('job_placement_model.pkl', 'rb') as file:
        model_stuff = pickle.load(file)
    model = model_stuff['model']
    print("Model successfully load ho gaya!")
except Exception as e:
    print(f"Error loading pickle file: {e}")

class StudentData(BaseModel):
    gender: int
    ssc_percentage: float
    hsc_percentage: float
    hsc_subject: int
    degree_percentage: float
    undergrad_degree: int
    work_experience: int
    emp_test_percentage: float
    specialisation: int
    mba_percent: float

# 1. Home Route par ab HTML Frontend dikhega
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"<h3>index.html file nahi mili! Error: {e}</h3>"


@app.post("/predict")
def predict_placement(student: StudentData):
    # Student data ko list me convert karna usi order me jisme train kiya tha
    features = [
        student.gender,
        student.ssc_percentage,
        student.hsc_percentage,
        student.hsc_subject,
        student.degree_percentage,
        student.undergrad_degree,
        student.work_experience,
        student.emp_test_percentage,
        student.specialisation,
        student.mba_percent
    ]
    
    # Model ko input dene ke liye 2D array chahiye [[data]]
    features_array = np.array([features])
    
    # Prediction lagana
    prediction = model.predict(features_array)
    
    
    pred_value = str(prediction[0]).strip()
    
    # Agar model string 'Placed' de raha ho ya numeric 1 de raha ho
    if pred_value == "Placed" or pred_value == "1":
        result = "Placed"
        raw_out = 1
    else:
        result = "Not Placed"
        raw_out = 0
    
    return {
        "status": "success",
        "prediction": result,
        "raw_output": raw_out  
    }