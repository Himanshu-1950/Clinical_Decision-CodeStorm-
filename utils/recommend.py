def recommend(disease):
    data = {
        "Diabetes": ["HbA1c","Fasting Sugar"],
        "Heart Disease": ["ECG","Lipid Profile"],
        "Flu": ["CBC"],
        "Cold": []
    }
    return data.get(disease, [])
   # Diagnostic test recommendation logic
  