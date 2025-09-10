import joblib
from lime.lime_text import LimeTextExplainer

class LimeWrapper:
    def __init__(self, pipeline, class_names=("ham","phish")):
        self.pipeline = pipeline
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    def explain(self, text, num_features=10):
        exp = self.explainer.explain_instance(text, self.predict_proba, num_features=num_features)
        return exp
