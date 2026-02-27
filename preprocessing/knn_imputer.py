# backend/preprocessing/knn_imputer.py

import pandas as pd
from sklearn.impute import KNNImputer as SklearnKNNImputer


class KNNImputer:
    def __init__(self, neighbors: int = 5, distanceMetric: str = "nan_euclidean"):
        # Attributes (as per class diagram)
        self.neighbors = neighbors
        self.distanceMetric = distanceMetric
        self.imputer = SklearnKNNImputer(
            n_neighbors=self.neighbors,
            metric=self.distanceMetric
        )

    # +fit(dataset : DataFrame) : void
    def fit(self, dataset: pd.DataFrame):
        """
        Fits the KNN imputer on dataset
        """
        self.imputer.fit(dataset)

    # +transform(dataset : DataFrame) : DataFrame
    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms dataset using fitted KNN imputer
        """
        imputed_array = self.imputer.transform(dataset)
        return pd.DataFrame(imputed_array, columns=dataset.columns)

    # +impute(dataset : DataFrame) : DataFrame
    def impute(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms dataset in one step
        """
        imputed_array = self.imputer.fit_transform(dataset)
        return pd.DataFrame(imputed_array, columns=dataset.columns)
