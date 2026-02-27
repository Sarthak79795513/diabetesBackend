# backend/models/risk_categorizer.py

class RiskCategorizer:
    def __init__(self):
        # Attributes (as per class diagram)
        self.thresholds = {
            "LOW": 0.3,
            "MEDIUM": 0.6
        }
        self.riskLabels = ["LOW", "MEDIUM", "HIGH"]

    # +categorizeRisk(score : float) : String
    def categorizeRisk(self, score: float) -> str:
        """
        Categorizes diabetes risk based on prediction score
        """
        if score < self.thresholds["LOW"]:
            return "LOW"
        elif score < self.thresholds["MEDIUM"]:
            return "MEDIUM"
        else:
            return "HIGH"

    # +adjustThresholds(newThresholds : dict) : void
    def adjustThresholds(self, newThresholds: dict):
        """
        Allows dynamic adjustment of risk thresholds
        """
        self.thresholds.update(newThresholds)
