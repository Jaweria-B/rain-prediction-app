import streamlit as st
import pandas as pd

from keras.models import load_model

# Load the trained model from the file
model = load_model('rain_model.h5')


# Add additional information about the dataset
st.write("""
# Rain Prediction App ☔️

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

# Get user input features
df = user_input_features()
# print(df.dtypes)

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

# Display the picture
st.image('./assets/clouds-sun-laughing.png', caption='Epic Battle: Sun vs. Clouds', use_column_width=True)

# Model Evaluation Section
st.write("# Model Evaluation")
st.write("The model underwent rigorous training using cutting-edge deep learning techniques and underwent comprehensive evaluation using a suite of performance metrics including accuracy, precision, recall, and F1-score.")

# Utilization of Deep Learning Techniques
st.write("### Utilization of Deep Learning Techniques")
st.write("The model leveraged state-of-the-art deep learning methodologies for both training and evaluation phases, ensuring it could capture intricate patterns within the data effectively.")

# Evaluation Metrics
st.write("### Evaluation Metrics")
st.write("Performance metrics such as accuracy, precision, recall, and F1-score were meticulously calculated to assess the model's predictive capabilities comprehensively.")

# Preprocessing Steps
st.write("### Preprocessing Steps")
st.write("The dataset underwent extensive preprocessing steps to ensure data quality and enhance model performance, including handling missing values, encoding categorical variables, and scaling numerical features.")

# Hyperparameter Tuning and Cross-Validation
st.write("### Hyperparameter Tuning and Cross-Validation")
st.write("Hyperparameters were fine-tuned iteratively, and cross-validation techniques were employed to optimize the model's performance.")

# Robust Performance
st.write("### Robust Performance")
st.write("The final trained model exhibited robust performance on unseen data, demonstrating high accuracy in predicting rain occurrences with confidence.")

# Understanding Input Features Section
st.write("# Understanding Input Features")
st.write("Here's a breakdown of some key features and their transformations:")

# Temporal Features
st.write("### Temporal Features")
st.write("Year, month, and day features were transformed into cyclic representations using sine and cosine transformations, allowing the model to capture seasonal and periodic patterns more effectively.")

# Location
st.write("### Location")
st.write("Geographic location plays a crucial role in weather patterns. By including location data, the model can adapt its predictions based on regional climate variations, enhancing its accuracy and relevance.")

# Weather Parameters
st.write("### Weather Parameters")
st.write("Weather parameters such as temperature, rainfall, humidity, etc., underwent extensive preprocessing techniques to ensure consistency and reliability.")

# RainToday
st.write("### RainToday")
st.write("This binary feature indicates whether it rained on the current day, enabling the model to learn from past weather occurrences and adjust its predictions accordingly.")

# Wind Directions and Speeds
st.write("### Wind Directions and Speeds")
st.write("Wind direction and speed data provide valuable insights into airflow patterns, enhancing the model's predictive capabilities for various weather conditions.")

# Conclusion
st.write("# Conclusion")
st.write("The Rain Prediction App combines deep learning techniques with meticulous preprocessing to deliver accurate weather forecasts. Further enhancements aim to improve predictive capabilities and user experience, empowering individuals and communities to make informed decisions in changing weather conditions.")

st.write(
"""
---
Made By **_Jaweria Batool_**
"""
)

# link to GitHub README file
st.write("For more information about how the app works, please check out the [GitHub README](https://github.com/Jaweria-B/rain-prediction-app) file.")
