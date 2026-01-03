import numpy as np

def top_diseases(model, X, encoder, n=3):
    probs = model.predict_proba(X)[0]
    idx = probs.argsort()[::-1][:n]

    return [
        {
            "disease": encoder.inverse_transform([i])[0],
            "probability": round(probs[i]*100,2)
        }
        for i in idx
    ]
