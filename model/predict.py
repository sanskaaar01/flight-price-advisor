import pandas as pd
import pickle

from model.preprocess import preprocess_data

# ---------- LOAD MODEL ----------
model = pickle.load(open("model/model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

def predict_price(input_dict):
    # ---------- CONVERT TO DF ----------
    df = pd.DataFrame([input_dict])

    # ---------- PREPROCESS ----------
    df = preprocess_data(df)

    # ---------- MATCH COLUMNS ----------
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    # ---------- PREDICT ----------
    price = model.predict(df)[0]

    return round(price, 2)

if __name__ == "__main__":
    sample_input = {
        "Airline": "IndiGo",
        "Date_of_Journey": "24/03/2019",
        "Source": "Delhi",
        "Destination": "Cochin",
        "Route": "DEL → COK",
        "Dep_Time": "22:20",
        "Arrival_Time": "01:10",
        "Duration": "2h 50m",
        "Total_Stops": "non-stop",
        "Additional_Info": "No info"
    }

    print("Predicted Price:", predict_price(sample_input))