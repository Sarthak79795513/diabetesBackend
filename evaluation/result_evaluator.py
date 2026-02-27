# backend/evaluation/result_evaluator.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ResultEvaluator:
    def __init__(self):
        # Attributes (as per class diagram)
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1Score = 0.0
        self.riskLevel = ""

    # +evaluatePerformance(y_true, y_pred) : void
    def evaluatePerformance(self, y_true, y_pred):
        """
        Evaluates model performance using standard metrics
        """
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1Score = f1_score(y_true, y_pred)

    # +calculateRiskLevel(score : float) : String
    def calculateRiskLevel(self, score: float) -> str:
        """
        Converts score to risk level (fallback if needed)
        """
        if score < 0.3:
            self.riskLevel = "LOW"
        elif score < 0.6:
            self.riskLevel = "MEDIUM"
        else:
            self.riskLevel = "HIGH"

        return self.riskLevel

    # +generateEvaluationGraphs() : void
    def generateEvaluationGraphs(self):
        """
        Placeholder for accuracy/loss/ROC graphs
        (handled by VisualizationModule)
        """
        pass

    # +displayFinalResult(patient, riskLevel) : void
    def displayFinalResult(self, patient, riskLevel: str):
        """
        Displays final prediction result
        """
        print("Patient Name:", patient.name)
        print("Predicted Diabetes Risk Level:", riskLevel)
