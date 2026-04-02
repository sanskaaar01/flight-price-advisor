from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model.predict import predict_price

app = FastAPI()

# ---------- INPUT SCHEMA ----------
class FlightInput(BaseModel):
    Airline: str
    Date_of_Journey: str
    Source: str
    Destination: str
    Route: str
    Dep_Time: str
    Arrival_Time: str
    Duration: str
    Total_Stops: str
    Additional_Info: str


# ---------- HOME ----------
@app.get("/")
def home():
    return {"message": "Flight Price API Running"}


# ---------- PREDICT ----------
@app.post("/predict")
def predict(data: FlightInput):
    input_dict = data.dict()

    price = predict_price(input_dict)

    return {
        "predicted_price": price,
        "recommendation": "Buy Now" if price < 6000 else "Wait",
        "best_day": "Tuesday"
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#o