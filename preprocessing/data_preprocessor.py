import pandas as pd
import joblib

class DataPreprocessor:
    def __init__(self):
        self.imputer = joblib.load("saved_models/imputer.pkl")
        self.scaler = joblib.load("saved_models/scaler.pkl")

    def preprocessInputData(self, patientData: dict) -> list:
        numericData = {
            "Pregnancies": patientData["pregnancies"],
            "Glucose": patientData["glucose"],
            "BloodPressure": patientData["bloodPressure"],
            "SkinThickness": patientData["skinThickness"],
            "Insulin": patientData["insulin"],
            "BMI": patientData["BMI"],
            "DiabetesPedigreeFunction": patientData["diabetesPedigreeFunction"],
            "Age": patientData["age"]
        }

        df = pd.DataFrame([numericData])

        # 🔥 ONLY transform (NO fit)
        df = self.imputer.transform(df)
        df = self.scaler.transform(df)

        return df[0].tolist()
