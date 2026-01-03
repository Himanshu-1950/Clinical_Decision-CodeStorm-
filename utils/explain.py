def explain(model, features):
    return sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
# Model explainability using feature importance
