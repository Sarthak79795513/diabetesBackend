import numpy as np
import joblib

class TriEnsembleModel:
    def __init__(self):
        models = joblib.load("saved_models/tri_ensemble.pkl")
        self.rf = models["rf"]
        self.xgb = models["xgb"]
        self.et = models["et"]

    def combinePredictions(self, data: list) -> float:
        rf_prob = self.rf.predict_proba([data])[0][1]
        xgb_prob = self.xgb.predict_proba([data])[0][1]
        et_prob = self.et.predict_proba([data])[0][1]

        return float(np.mean([rf_prob, xgb_prob, et_prob]))
