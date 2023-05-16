import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error
from PIL import Image

# Load the pickled model, encoder, and scaler
with open('xgb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('ohe.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('MinMaxScaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the function to preprocess the input data
def preprocess_data(df):
    df['hr'] = df['hr'].astype(str)
    df['weekday'] = df['weekday'].astype(str)
    df[['season','yr','hr','mnth','holiday','weekday','workingday','weathersit']] = df[['season','yr','hr','mnth','holiday','weekday','workingday','weathersit']].astype('object')
    num_cols = ['temp','hum','windspeed']
    cat_cols = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']
    cat_df = df[cat_cols]
    num_df = df[num_cols]
    cat_enc = encoder.transform(cat_df)
    num_scl = scaler.transform(num_df)
    df1 = pd.concat([pd.DataFrame(cat_enc,columns=encoder.get_feature_names_out(cat_cols)),pd.DataFrame(num_scl,columns=num_cols)],axis=1)
    return df1

# Define the function to predict bike rentals
def predict_bike_rentals(data):
    pred = model.predict(data)
    return pred


# Define the Streamlit app
def app():
    
    st.markdown(
        """
        <style>
        body {
            background-color: lightblue; /* Change the background color here */
        }
        .title {
            color: #0492C2; /* Change the color code here */
            font-size: 38px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .header {
            color: #0492C2; /* Change the color code here */
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subheader {
            color: #0492C2; /* Change the color code here */
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stApp main {
            background-color: lightblue; /* Change the background color here */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='title'>Bike Sharing Rental Prediction</h1>", unsafe_allow_html=True)
    image = Image.open("image.jpg")
    st.image(image)

    st.sidebar.markdown("<h2 class='header'>Bike Sharing Rental Project</h2>", unsafe_allow_html=True)
    image1 = Image.open("icon.jpg")
    st.sidebar.image(image1)
    st.sidebar.markdown("<h3 class='subheader'>Enter the details below to predict the number of bike rentals</h3>", unsafe_allow_html=True)
    
    season = st.sidebar.selectbox('Season', ['summer', 'winter', 'springer', 'fall'])
    year = st.sidebar.selectbox('Year', ['2011','2012'])
    month = st.sidebar.selectbox('Month',['1','2','3','4','5','6','7','8','9','10','11','12'])
    hour = st.sidebar.number_input('Hour',min_value=0, max_value=23, step=1)
    holiday = st.sidebar.selectbox('Holiday', ['No', 'Yes'])
    weekday = st.sidebar.number_input('Weekday', min_value=0, max_value=6, step=1)
    workingday = st.sidebar.selectbox('Working Day', ['No work', 'Working Day'])
    weather = st.sidebar.selectbox('Weather', ['Clear','Mist','Light Snow','Heavy Rain'])
    temp = st.sidebar.number_input('Temperature', min_value=0.0, max_value=50.0, step=0.1)
    humidity = st.sidebar.number_input('Humidity', min_value=0, max_value=100, step=1)
    windspeed = st.sidebar.number_input('Windspeed', min_value=0.0, max_value=50.0, step=0.1)
    df = pd.DataFrame({'season': [season], 'yr':[year], 'hr':[hour], 
                       'mnth':[month], 'holiday':[holiday],'weekday':[weekday],
                       'workingday': [workingday], 'weathersit':[weather], 
                       'temp': [temp], 'hum': [humidity], 'windspeed': [windspeed]})
    data = preprocess_data(df)
    if st.button('Predict'):
        pred = predict_bike_rentals(data)[0]
        st.success(f'The predicted number of bike rentals is {int(pred)}')

        df_rentals = pd.read_csv('bike_rent.csv')
        df_rentals.drop(['instant','dteday','casual','registered','atemp'],axis=1,inplace=True)
        df_rentals.replace('[~`!@#$%^&*()_+{}\[\]:;"\'<>,?/\\|]', np.nan, regex=True, inplace=True)
        df_rentals[['temp','hum','windspeed']] = df_rentals[['temp','hum','windspeed']].apply(pd.to_numeric)
        df_rentals['hr'] = df_rentals['hr'].astype(str)
        df_rentals['weekday'] = df_rentals['weekday'].astype(str)
        df_rentals[['temp',]] = df_rentals[['temp']].fillna(df_rentals[['temp']].mean()) 
        df_rentals[['hum','windspeed']] = df_rentals[['hum','windspeed']].fillna(df_rentals[['hum','windspeed']].median())
        cols = ['season','workingday','weathersit','yr','mnth','holiday']
        df_rentals[cols] = df_rentals[cols].fillna(df_rentals.mode().iloc[0])
        x = df_rentals.drop(['cnt'],axis=1)
        y = df_rentals['cnt']
        data2 = preprocess_data(x)
        pred2 = predict_bike_rentals(data2)

        # Calculate the rmse
        mse = mean_squared_error(y, pred2)
        rmse = np.sqrt(mse)

        # Plot the predicted and actual counts
        st.markdown("<h3 class='subheader'>Actual ve Predicted Graph of the ML Model Predicting the Count</h3>", unsafe_allow_html=True)
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y, y=pred2)

        # Add diagonal line
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.plot([xmin, xmax], [ymin, ymax], ls="--", c=".3")

        # Add RMSE value to plot
        plt.text(xmin, ymax, f"RMSE = {rmse:.2f}", ha='left', va='top', fontsize=12)

        # Set axis labels and title
        plt.xlabel("Actual values",fontweight='bold')
        plt.ylabel("Predicted values",fontweight='bold')
        plt.title("Actual vs Predicted values")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
        
if __name__ == '__main__':
    app()