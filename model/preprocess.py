import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # ---------- DROP NULL ----------
    df = df.dropna()

    # ---------- DATE FEATURES ----------
    df['Date_of_Journey'] = pd.to_datetime(
        df['Date_of_Journey'], format="%d/%m/%Y"
    )

    df['journey_day'] = df['Date_of_Journey'].dt.day
    df['journey_month'] = df['Date_of_Journey'].dt.month
    df['is_weekend'] = df['Date_of_Journey'].dt.weekday >= 5

    df.drop('Date_of_Journey', axis=1, inplace=True)

    # ---------- CLEAN TIME STRINGS ----------
    # Handles cases like "22:20 22 Mar"
    df['Dep_Time'] = df['Dep_Time'].astype(str).str.split().str[0]
    df['Arrival_Time'] = df['Arrival_Time'].astype(str).str.split().str[0]

    # ---------- TIME FEATURES ----------
    df['Dep_hour'] = pd.to_datetime(
        df['Dep_Time'], format="%H:%M"
    ).dt.hour

    df['Dep_min'] = pd.to_datetime(
        df['Dep_Time'], format="%H:%M"
    ).dt.minute

    df['Arrival_hour'] = pd.to_datetime(
        df['Arrival_Time'], format="%H:%M"
    ).dt.hour

    df['Arrival_min'] = pd.to_datetime(
        df['Arrival_Time'], format="%H:%M"
    ).dt.minute

    df.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

    # ---------- DURATION ----------
    def convert_duration(x):
        x = str(x).split()
        hours = 0
        mins = 0

        for i in x:
            if 'h' in i:
                hours = int(i.replace('h', ''))
            elif 'm' in i:
                mins = int(i.replace('m', ''))

        return hours * 60 + mins

    df['duration_mins'] = df['Duration'].apply(convert_duration)
    df.drop('Duration', axis=1, inplace=True)

    # ---------- TOTAL STOPS ----------
    df['Total_Stops'] = df['Total_Stops'].map({
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3,
        '4 stops': 4
    })

    # ---------- DROP USELESS ----------
    df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

    # ---------- ENCODING ----------
    df = pd.get_dummies(
        df,
        columns=['Airline', 'Source', 'Destination'],
        drop_first=True
    )

    # ---------- WEATHER ----------
    df['weather'] = 0  # placeholder for API later

    return df