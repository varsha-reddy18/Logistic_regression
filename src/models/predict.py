import joblib
import numpy as np

model = joblib.load(
    "../../saved_models/logistic_model.pkl"
)

scaler = joblib.load(
    "../../saved_models/scaler.pkl"
)

def predict_survival(
    pclass,
    sex,
    age,
    fare,
    embarked
):

    data = np.array([[
        pclass,
        sex,
        age,
        fare,
        embarked
    ]])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)

    return prediction[0]