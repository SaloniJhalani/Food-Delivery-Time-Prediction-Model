from pathlib import Path
import datetime

import pandas as pd
import streamlit as st

import main
import predict


def get_user_input(df_train):
    st.sidebar.write(f"**Order Related Information**")
    date = st.sidebar.date_input("what is the Order Date?")
    order_time = st.sidebar.time_input("What is the Order Time?", step=60)
    order_datetime = datetime.datetime.combine(date, order_time)
    pickup_time = st.sidebar.time_input("What is the Order Pickup Time?",
                                        order_datetime + datetime.timedelta(minutes=15), step=60)
    order_type = st.sidebar.selectbox('What is the type of order?',
                                      df_train['Type_of_order'].unique())
    multiple_deliveries = st.sidebar.selectbox('How many deliveries are combined?',
                                               sorted(df_train['multiple_deliveries'].unique().astype('int')))
    st.sidebar.write(f"**Location Related Information**")
    restaurant_latitude = st.sidebar.text_input("What is the restaurant latitude?", "14.829222")
    restaurant_longitude = st.sidebar.text_input("What is the restaurant longitude?", "67.920922")
    delivery_location_latitude = st.sidebar.text_input("What is the delivery location latitude?", "14.929222")
    delivery_location_longitude = st.sidebar.text_input("What is the delivery location longitude?", "68.860922")
    st.sidebar.write(f"**Delivery Person Related Information**")
    delivery_person_age = st.sidebar.slider("How old is the delivery person?",
                                            int(df_train['Delivery_person_Age'].min()),
                                            int(df_train['Delivery_person_Age'].max()),
                                            int(df_train['Delivery_person_Age'].mean()))
    delivery_person_rating = st.sidebar.slider("What is delivery person rating?",
                                               float(df_train['Delivery_person_Ratings'].min()),
                                               float(df_train['Delivery_person_Ratings'].max()),
                                               float(df_train['Delivery_person_Ratings'].mean()))
    vehicle = st.sidebar.selectbox('What type of vehicle delivery person has?',
                                   df_train['Type_of_vehicle'].unique())
    vehicle_condition = st.sidebar.selectbox('What is the Vehicle condition of delivery person?',
                                             sorted(df_train['Vehicle_condition'].unique()))
    st.sidebar.write(f"**City Related Information**")
    city_code = st.sidebar.selectbox('What is the city name of delivery?',
                                     df_train['City_code'].unique())
    city = st.sidebar.selectbox('Which type of city it is?',
                                df_train['City'].unique())
    st.sidebar.write(f"**Weather Conditions/Event Related Information**")
    road_density = st.sidebar.selectbox('What is road traffic density?',
                                        df_train['Road_traffic_density'].unique())
    weather_conditions = st.sidebar.selectbox('How is the weather?',
                                              df_train['Weather_conditions'].unique())
    festival = st.sidebar.selectbox('Is there a festival?',
                                    df_train['Festival'].unique())

    X = pd.DataFrame({
        'ID': '123456',
        'Delivery_person_ID': city_code + 'RES13DEL02',
        'Delivery_person_Age': delivery_person_age,
        'Delivery_person_Ratings': delivery_person_rating,
        'Restaurant_latitude': format(float(restaurant_latitude), ".6f"),
        'Restaurant_longitude': format(float(restaurant_longitude), ".6f"),
        'Delivery_location_latitude': format(float(delivery_location_latitude), ".6f"),
        'Delivery_location_longitude': format(float(delivery_location_longitude), ".6f"),
        'Order_Date': date.strftime('%d-%m-%Y'),
        'Time_Orderd': order_time.strftime('%H:%M:%S'),
        'Time_Order_picked': pickup_time.strftime('%H:%M:%S'),
        'Weatherconditions': 'conditions ' + weather_conditions,
        'Road_traffic_density': road_density,
        'Vehicle_condition': vehicle_condition,
        'Type_of_order': order_type,
        'Type_of_vehicle': vehicle,
        'multiple_deliveries': multiple_deliveries,
        'Festival': festival,
        'City': city
    }, index=[0])
    return X


if __name__ == "__main__":
    st.set_page_config(page_title="Food Delivery Time Prediction", page_icon=None, layout="centered",
                       initial_sidebar_state="auto")

    # Read in training data
    df_train = pd.read_csv(str(Path(__file__).parents[1] / 'data/train.csv'))
    main.cleaning_steps(df_train)

    # Displaying text
    st.title("Food Delivery Time Prediction")
    # Displaying an image
    st.image(str(Path(__file__).parents[1] / 'img/food-delivery.png'), width=700)

    st.write("""  
            The food delivery time prediction model is vital in ensuring prompt and accurate delivery in the food delivery industry. Leveraging advanced data cleaning techniques and feature engineering, a robust food delivery time prediction model is developed.
            
            This model predicts food delivery time based on a range of factors, including order details, location, city, delivery person information, and weather conditions.  
             """)

    ##create the sidebar
    st.sidebar.header("User Input Parameters")

    ##create function for User input

    input_df = get_user_input(df_train)  # get user input from sidebar

    order_date = input_df['Order_Date'][0]
    order_time = input_df['Time_Orderd'][0]
    order_date_time = datetime.datetime.strptime(f'{order_date} {order_time}', '%d-%m-%Y %H:%M:%S')
    order_pickup_time = input_df['Time_Order_picked'][0]
    order_pickup_date_time = datetime.datetime.strptime(f'{order_date} {order_pickup_time}', '%d-%m-%Y %H:%M:%S')

    total_delivery_minutes = round(predict.predict(input_df)[0], 2)  # get predicitions
    minutes = int(total_delivery_minutes)
    seconds = int((total_delivery_minutes - minutes) * 60)
    X = order_pickup_date_time + datetime.timedelta(minutes=minutes, seconds=seconds)

    # display predictions
    st.subheader("Order Details")
    st.write(f"**Order was Placed on :** {order_date_time}")
    st.write(f"**Order was Picked up at :** {order_pickup_date_time}")
    st.subheader("Prediction")
    formatted_X = "{:.2f}".format(total_delivery_minutes)
    st.write(f"**Total Delivery Time is :** {formatted_X} mins")
    st.write(f"**Order will be delivered by :** {X}")
