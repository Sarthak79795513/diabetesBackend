# backend/preprocessing/feature_selector.py

import pandas as pd

class FeatureSelector:
    def __init__(self):
        # Attributes (as per class diagram)
        self.selectedFeatures = []

    # +selectImportantFeatures(dataset : DataFrame) : list
    def selectImportantFeatures(self, dataset: pd.DataFrame) -> list:
        """
        Selects important features based on correlation with target.
        (Simple + explainable approach, can be extended later)
        """
        if "Outcome" not in dataset.columns:
            self.selectedFeatures = dataset.columns.tolist()
            return self.selectedFeatures

        correlation = dataset.corr()["Outcome"].abs()
        self.selectedFeatures = correlation[correlation > 0.2].index.tolist()

        if "Outcome" in self.selectedFeatures:
            self.selectedFeatures.remove("Outcome")

        return self.selectedFeatures

    # +reduceDimensionality(dataset : DataFrame) : DataFrame
    def reduceDimensionality(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces dataset to selected important features
        """
        if not self.selectedFeatures:
            return dataset
        return dataset[self.selectedFeatures]
