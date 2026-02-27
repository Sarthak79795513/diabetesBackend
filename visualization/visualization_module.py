import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class VisualizationModule:
    def __init__(self):
        # Attributes (as per class diagram)
        self.featureImportanceChart = None
        self.correlationHeatmap = None

    # +plotFeatureImportance(model : TriEnsembleModel) : void
    def plotFeatureImportance(self, model):
        """
        Plots feature importance using Random Forest model
        """

        # RandomForest returns list, not dict
        importance = model.getFeatureImportance()

        if importance is None or len(importance) == 0:
            print("No feature importance available")
            return

        # 🔑 Feature names must match PIMA dataset order
        features = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        values = importance

        plt.figure(figsize=(8, 5))
        plt.barh(features, values, color="skyblue")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.show()

    # +displayRiskVisualization(riskLevel : String) : void
    def displayRiskVisualization(self, riskLevel: str):
        """
        Displays simple visual indicator for risk level
        """

        colors = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red"
        }

        plt.figure(figsize=(4, 4))
        plt.text(
            0.5,
            0.5,
            riskLevel,
            fontsize=22,
            ha="center",
            va="center",
            color=colors.get(riskLevel, "black"),
            weight="bold"
        )
        plt.axis("off")
        plt.title("Diabetes Risk Level")
        plt.show()

    # +showCorrelationHeatmap(dataset : DataFrame) : void
    def showCorrelationHeatmap(self, dataset: pd.DataFrame):
        """
        Displays correlation heatmap of dataset
        """

        if dataset is None or dataset.empty:
            print("Dataset is empty, cannot plot heatmap")
            return

        plt.figure(figsize=(10, 6))
        sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()
