# backend/main.py

from models.patient import Patient
from preprocessing.data_preprocessor import DataPreprocessor
from models.tri_ensemble_model import TriEnsembleModel
from models.risk_categorizer import RiskCategorizer
from database.database_manager import Database
from reports.report_generator import ReportGenerator
from visualization.visualization_module import VisualizationModule


def main():
    print("=== Diabetes Risk Prediction System ===")

    preprocessor = DataPreprocessor()
    model = TriEnsembleModel()
    riskCategorizer = RiskCategorizer()
    database = Database()
    reportGenerator = ReportGenerator()
    visualizer = VisualizationModule()

    # Sample input
    inputData = {
        "userID": 1,
        "name": "Test User",
        "age": 35,
        "gender": "Male",
        "pregnancies": 0,
        "glucose": 148,
        "bloodPressure": 72,
        "skinThickness": 35,
        "insulin": 0,
        "BMI": 33.6,
        "diabetesPedigreeFunction": 0.627
    }

    # Create Patient
    patient = Patient()
    patient.inputPatientData(inputData)

    print("Patient data received")

    # 🔑 Convert Patient → dict
    patientDict = patient.displayPatientData()

    # 🔑 Preprocess ONLY numeric data
    processedData = preprocessor.preprocessInputData(patientDict)

    print("Data preprocessing completed")

    # Prediction
    score = model.combinePredictions(processedData)
    riskLevel = riskCategorizer.categorizeRisk(score)

    print("Risk Level:", riskLevel)
    print("Risk Score:", round(score, 3))

    # Save DB
    database.savePatientData(patient, riskLevel)

    # Reports
    reportGenerator.generatePatientReport(patient, riskLevel)
    reportGenerator.exportReport("PDF")
    reportGenerator.exportReport("HTML")

    # Visualization
    visualizer.displayRiskVisualization(riskLevel)

    print("=== Process Completed ===")


if __name__ == "__main__":
    main()
