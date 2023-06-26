from pathlib import Path
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def update_column_name(df):
    df.rename(columns={'Weatherconditions': 'Weather_conditions'}, inplace=True)


def extract_feature_value(df):
    # Extract Weather conditions
    df['Weather_conditions'] = df['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip())
    # Extract city code from Delivery person ID
    df['City_code'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]

    #Remove Whitespaces on categorical value
    categorical_columns = df.select_dtypes(include='object').columns
    for column in categorical_columns:
        df[column] = df[column].str.strip()


def extract_label_value(df):
    # Extract time and convert to int
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))


def drop_columns(df):
    df.drop(['ID', 'Delivery_person_ID'], axis=1, inplace=True)


def update_datatype(df):
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')
    df['multiple_deliveries'] = df['multiple_deliveries'].astype('float64')
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%d-%m-%Y")


def convert_nan(df):
    df.replace('NaN', float(np.nan), regex=True, inplace=True)


def handle_null_values(df):
    df['Delivery_person_Age'].fillna(np.random.choice(df['Delivery_person_Age']), inplace=True)
    df['Weather_conditions'].fillna(np.random.choice(df['Weather_conditions']), inplace=True)
    df['City'].fillna(df['City'].mode()[0], inplace=True)
    df['Festival'].fillna(df['Festival'].mode()[0], inplace=True)
    df['multiple_deliveries'].fillna(df['multiple_deliveries'].mode()[0], inplace=True)
    df['Road_traffic_density'].fillna(df['Road_traffic_density'].mode()[0], inplace=True)
    df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)


def extract_date_features(data):
    data["day"] = data.Order_Date.dt.day
    data["month"] = data.Order_Date.dt.month
    data["quarter"] = data.Order_Date.dt.quarter
    data["year"] = data.Order_Date.dt.year
    data['day_of_week'] = data.Order_Date.dt.day_of_week.astype(int)
    data["is_month_start"] = data.Order_Date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.Order_Date.dt.is_month_end.astype(int)
    data["is_quarter_start"] = data.Order_Date.dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data.Order_Date.dt.is_quarter_end.astype(int)
    data["is_year_start"] = data.Order_Date.dt.is_year_start.astype(int)
    data["is_year_end"] = data.Order_Date.dt.is_year_end.astype(int)
    data['is_weekend'] = np.where(data['day_of_week'].isin([5, 6]), 1, 0)


def calculate_time_diff(df):
    # Find the difference between ordered time & picked time
    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])

    df['Time_Order_picked_formatted'] = df['Order_Date'] + np.where(df['Time_Order_picked'] < df['Time_Orderd'],
                                                                    pd.DateOffset(days=1), pd.DateOffset(days=0)) + df[
                                            'Time_Order_picked']
    df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']

    df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df[
        'Time_Ordered_formatted']).dt.total_seconds() / 60

    # Handle null values by filling with the median
    df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)

    # Drop all the time & date related columns
    df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'],
            axis=1, inplace=True)


def calculate_distance(df):
    df['distance'] = np.zeros(len(df))
    restaurant_coordinates = df[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
    delivery_location_coordinates = df[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
    df['distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in
                               zip(restaurant_coordinates, delivery_location_coordinates)])
    df['distance'] = df['distance'].astype("str").str.extract('(\d+)').astype("int64")


def label_encoding(df):
    categorical_columns = df.select_dtypes(include='object').columns
    label_encoders = {}

    # Iterate over each categorical column and fit a label encoder
    for column in categorical_columns:
        df[column] = df[column].str.strip()  # Remove whitespaces
        label_encoder = LabelEncoder()
        label_encoder.fit(df[column])
        df[column] = label_encoder.transform(df[column])
        label_encoders[column] = label_encoder
    return label_encoders


def data_split(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    scaler = StandardScaler()

    # Fit the scaler on the training data
    scaler.fit(X_train)

    # Perform standardization
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def cleaning_steps(df):
    update_column_name(df)
    extract_feature_value(df)
    drop_columns(df)
    update_datatype(df)
    convert_nan(df)
    handle_null_values(df)

def perform_feature_engineering(df):
    extract_date_features(df)
    calculate_time_diff(df)
    calculate_distance(df)


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("Mean Squared Error (MSE):", round(mse, 2))
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))
    print("R-squared (R2) Score:", round(r2, 2))


if __name__ == "__main__":
    df_train = pd.read_csv(str(Path(__file__).parents[1] / 'data/train.csv'))  # Load Data
    cleaning_steps(df_train)  # Perform Cleaning
    extract_label_value(df_train) #Extract Label Value
    perform_feature_engineering(df_train)  # Perform feature engineering

    # Split features & label
    X = df_train.drop('Time_taken(min)', axis=1)  # Features
    y = df_train['Time_taken(min)']  # Target variable

    label_encoders = label_encoding(X)  # Label Encoding
    X_train, X_test, y_train, y_test = data_split(X, y)  # Test Train Split
    X_train, X_test, scaler = standardize(X_train, X_test)  # Standardization

    # Build Model
    model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    # Save Model
    with open(str(Path(__file__).parents[1] / 'code/model.pickle'), 'wb') as f:
        pickle.dump((model, label_encoders, scaler), f)
