import streamlit as st
import pandas as pd

from tensorflow import keras
from keras.models import load_model

# Load the trained model from the file
model = load_model('rain_model.keras')


# Add additional information about the dataset
st.write("""
# Rain Prediction App

This app predicts whether it will rain tomorrow based on various weather parameters. \n
🌦️ **Predict Rain, Stay Dry, and Enjoy the Weather!** 🌈
""")

st.write('---')

# Load the dataset from the CSV file
df = pd.read_csv('processed_rain_data.csv')

X = df.drop(["RainTomorrow"], axis=1)
y = df["RainTomorrow"]

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Location = st.sidebar.slider('Location', X.Location.min(), X.Location.max(), X.Location.mean())
    MinTemp = st.sidebar.slider('MinTemp', X.MinTemp.min(), X.MinTemp.max(), X.MinTemp.mean())
    MaxTemp = st.sidebar.slider('MaxTemp', X.MaxTemp.min(), X.MaxTemp.max(), X.MaxTemp.mean())
    Rainfall = st.sidebar.slider('Rainfall', X.Rainfall.min(), X.Rainfall.max(), X.Rainfall.mean())
    Evaporation = st.sidebar.slider('Evaporation', X.Evaporation.min(), X.Evaporation.max(), X.Evaporation.mean())
    Sunshine = st.sidebar.slider('Sunshine', X.Sunshine.min(), X.Sunshine.max(), X.Sunshine.mean())
    WindGustDir = st.sidebar.selectbox('WindGustDir', X.WindGustDir.unique())
    WindGustSpeed = st.sidebar.slider('WindGustSpeed', X.WindGustSpeed.min(), X.WindGustSpeed.max(), X.WindGustSpeed.mean())
    WindDir9am = st.sidebar.selectbox('WindDir9am', X.WindDir9am.unique())
    WindDir3pm = st.sidebar.selectbox('WindDir3pm', X.WindDir3pm.unique())
    WindSpeed9am = st.sidebar.slider('WindSpeed9am', X.WindSpeed9am.min(), X.WindSpeed9am.max(), X.WindSpeed9am.mean())
    WindSpeed3pm = st.sidebar.slider('WindSpeed3pm', X.WindSpeed3pm.min(), X.WindSpeed3pm.max(), X.WindSpeed3pm.mean())
    Humidity9am = st.sidebar.slider('Humidity9am', X.Humidity9am.min(), X.Humidity9am.max(), X.Humidity9am.mean())
    Humidity3pm = st.sidebar.slider('Humidity3pm', X.Humidity3pm.min(), X.Humidity3pm.max(), X.Humidity3pm.mean())
    Pressure9am = st.sidebar.slider('Pressure9am', X.Pressure9am.min(), X.Pressure9am.max(), X.Pressure9am.mean())
    Pressure3pm = st.sidebar.slider('Pressure3pm', X.Pressure3pm.min(), X.Pressure3pm.max(), X.Pressure3pm.mean())
    Cloud9am = st.sidebar.slider('Cloud9am', X.Cloud9am.min(), X.Cloud9am.max(), X.Cloud9am.mean())
    Cloud3pm = st.sidebar.slider('Cloud3pm', X.Cloud3pm.min(), X.Cloud3pm.max(), X.Cloud3pm.mean())
    Temp9am = st.sidebar.slider('Temp9am', X.Temp9am.min(), X.Temp9am.max(), X.Temp9am.mean())
    Temp3pm = st.sidebar.slider('Temp3pm', X.Temp3pm.min(), X.Temp3pm.max(), X.Temp3pm.mean())
    RainToday = st.sidebar.selectbox('RainToday', ['Yes', 'No'])
    # Handle RainToday encoding (one-hot encoding)
    RainToday_encoded = 1 if RainToday == 'Yes' else 0
    year = st.sidebar.slider('Year', X.year.min(), X.year.max(), X.year.mean())
    month_sin = st.sidebar.slider('Month_sin', X.month_sin.min(), X.month_sin.max(), X.month_sin.mean())
    month_cos = st.sidebar.slider('Month_cos', X.month_cos.min(), X.month_cos.max(), X.month_cos.mean())
    day_sin = st.sidebar.slider('Day_sin', X.day_sin.min(), X.day_sin.max(), X.day_sin.mean())
    day_cos = st.sidebar.slider('Day_cos', X.day_cos.min(), X.day_cos.max(), X.day_cos.mean())

    
    # Create a dictionary to hold the user input features
    data = {'Location': Location,
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'Evaporation': Evaporation,
        'Sunshine': Sunshine,
        'WindGustDir': WindGustDir,
        'WindGustSpeed': WindGustSpeed,
        'WindDir9am': WindDir9am,
        'WindDir3pm': WindDir3pm,
        'WindSpeed9am': WindSpeed9am,
        'WindSpeed3pm': WindSpeed3pm,
        'Humidity9am': Humidity9am,
        'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am,
        'Pressure3pm': Pressure3pm,
        'Cloud9am': Cloud9am,
        'Cloud3pm': Cloud3pm,
        'Temp9am': Temp9am,
        'Temp3pm': Temp3pm,
        'RainToday': RainToday_encoded,
        'Year': year,
        'Month_sin': month_sin,
        'Month_cos': month_cos,
        'Day_sin': day_sin,
        'Day_cos': day_cos}

    return pd.DataFrame(data, index=[0])

# Assuming 'data' is the DataFrame containing your rain predictor data
# Modify 'data' as per your actual DataFrame
# Get user input features
df = user_input_features()
print(df.dtypes)
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Apply Model to Make Prediction
y_pred = model.predict(df)
y_pred = (y_pred > 0.5)

# Convert y_pred to a human-readable string
prediction = "Rain Tomorrow" if y_pred[0][0] else "No Rain Tomorrow"

# Display Prediction DataFrame
st.header('Prediction of Rain Tomorrow')
prediction_df = pd.DataFrame({"Prediction": [y_pred[0]]})
st.dataframe(prediction_df)

# Display Prediction Statement with styling
st.write('---')
st.header('Prediction Statement')
st.markdown(f"""
            * <p style='font-size:24px; font-style:italic;'>{prediction}</p>""", unsafe_allow_html=True)

st.divider()

# Model Evaluation
st.write("## Model Evaluation")
st.write("The model was trained using deep learning techniques and evaluated using various performance metrics such as accuracy, precision, recall, and F1-score. Key points regarding model evaluation include:")

st.write("- Utilization of deep learning techniques for training and evaluation.")
st.write("- Evaluation metrics include accuracy, precision, recall, and F1-score.")
st.write("- Extensive preprocessing steps were conducted on the dataset before training, including:")
st.write("  - Handling missing values.")
st.write("  - Encoding categorical variables.")
st.write("  - Scaling numerical features.")
st.write("- Hyperparameter tuning and cross-validation techniques were employed to optimize model performance.")
st.write("- The final trained model demonstrates robust performance on unseen data and exhibits high accuracy in predicting rain occurrences.")

# Conclusion
st.write("""
## Conclusion

This Rain Prediction App serves as a powerful tool for weather forecasting based on historical data. Key points about the conclusion include:

- Each feature in the dataset underwent meticulous modification during preprocessing, ensuring that the model captures the most relevant information for prediction.
- The application showcases the efficacy of deep learning in analyzing complex patterns within weather data and making accurate predictions about future rain occurrences.
- Moving forward, potential enhancements could include:
  - Integrating real-time weather data streams for dynamic updates.
  - Enhancing the user interface to provide more interactive visualizations and insights.
""")


st.write(
"""
Made By **_Jaweria Batool_**
"""
)

# link to GitHub README file
st.write("For more information about how the app works, please check out the [GitHub README](https://github.com/Jaweria-B/rain-prediction-app) file.")
