# 🌟 Welcome to the Amazing Rain Predictor App! 🌧️

<p align=center>
<img src="./assets/cloud-sun-img.png" alt="Rain Predictor Logo" style="width: 400px;"/>
</p>

## 📖 About

The Rain Predictor App is an extraordinary tool that harnesses the power of cutting-edge machine learning algorithms to predict rain occurrences with remarkable accuracy. Developed by a team of passionate data scientists and weather enthusiasts, this app combines state-of-the-art technology with the latest weather data to provide you with real-time rain forecasts.

### 📁 Repository Structure

- **model_building**: This directory contains the code for training and evaluating the machine learning model used in the Rain Predictor App. The model is built using TensorFlow and Keras and is saved as a `.keras` file.
  
- **app.py**: This file is the main application script that implements the user interface using Streamlit. It allows users to input various parameters such as location, temperature, humidity, etc., and generates predictions using the trained model.

## ⚙️ Features

- **Accurate Rain Prediction**: Our advanced machine learning model accurately predicts rain occurrences based on historical weather data.

- **Interactive User Interface**: Enjoy a user-friendly interface designed to provide you with an immersive experience while exploring weather forecasts.

- **Customizable Parameters**: Tailor the predictions to your specific needs by adjusting parameters such as location, temperature, humidity, and more.

- **Real-time Updates**: Stay informed about changing weather conditions with real-time updates and notifications.

- **Visualizations**: Gain insights into weather patterns through stunning visualizations and graphs.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jaweria-B/rain-prediction-app.git
   ```

2. Navigate to the project directory:
   ```bash
   cd rain-prediction-app
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Live Demo 🚀

Check out the live demo [here](https://rain-prediction-jb.streamlit.app/) and experience the magic ✨ of predicting rain with our awesome app! 🌧️🔮


## 🚀 Getting Started

1. Once the app is running, you will be presented with the main dashboard.

2. Use the sidebar to customize the input parameters such as location, temperature, humidity, etc.

3. Click the "Predict" button to generate a rain prediction based on your selected parameters.

4. Explore the results and visualize the predicted rain occurrences.

## 🌐 Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- Pandas
- NumPy

## 📊 Dataset

The Rain Predictor App utilizes a comprehensive dataset containing historical weather data, including temperature, humidity, wind speed, and other relevant features.

# Model Evaluation 📊
The model underwent rigorous training using cutting-edge deep learning techniques and underwent comprehensive evaluation using a suite of performance metrics including accuracy, precision, recall, and F1-score. Here's a deeper dive into the model evaluation process:

## Utilization of Deep Learning Techniques 🧠
The model leveraged state-of-the-art deep learning methodologies for both training and evaluation phases, ensuring it could capture intricate patterns within the data effectively.

## Evaluation Metrics 📉
Performance metrics such as accuracy, precision, recall, and F1-score were meticulously calculated to assess the model's predictive capabilities comprehensively.

## Preprocessing Steps 🛠️
Prior to model training, the dataset underwent extensive preprocessing steps to ensure data quality and enhance model performance. These steps included:
- Handling Missing Values: Robust techniques were employed to address any missing data points, ensuring the integrity of the dataset.
- Encoding Categorical Variables: Categorical variables were encoded using advanced encoding techniques to transform them into a format suitable for deep learning algorithms.
- Scaling Numerical Features: Numerical features were appropriately scaled to ensure uniformity and facilitate convergence during model training.

## Hyperparameter Tuning and Cross-Validation 🎯
Hyperparameters were fine-tuned iteratively, and cross-validation techniques were employed to optimize the model's performance. This meticulous process ensured that the model achieved its maximum potential in terms of accuracy and generalization.

## Robust Performance ✔️
The final trained model exhibited robust performance on unseen data, demonstrating high accuracy in predicting rain occurrences with confidence.

# Understanding Input Features 🌐
In addition to model evaluation, it's crucial to understand the significance of each input feature in the dataset and how they were modified to enhance predictive capabilities. Here's a breakdown of some key features and their transformations:

## Temporal Features 📅
Originally represented as discrete values, year, month, and day features were transformed into cyclic representations using sine and cosine transformations. This transformation preserves the cyclic nature of time-related features, allowing the model to capture seasonal and periodic patterns more effectively.

## Location 🌍
Geographic location plays a crucial role in weather patterns. By including location data, the model can adapt its predictions based on regional climate variations, enhancing its accuracy and relevance.

## Weather Parameters ☔
Weather parameters such as temperature, rainfall, humidity, etc., underwent extensive preprocessing techniques to ensure consistency and reliability. This enables the model to make precise predictions based on nuanced weather patterns.

## RainToday 🌧️
This binary feature indicates whether it rained on the current day. By incorporating this information, the model can learn from past weather occurrences and adjust its predictions accordingly.

## Wind Directions and Speeds 💨
Wind patterns significantly influence weather phenomena. By including wind direction and speed data, the model gains valuable insights into airflow patterns, enhancing its predictive capabilities for various weather conditions.

# Conclusion 🚀
The Rain Prediction App combines cutting-edge deep learning techniques with meticulous preprocessing and insightful feature engineering to deliver accurate weather forecasts. As we continue to enhance the app, our goal is to further improve predictive capabilities and user experience, empowering individuals and communities to make informed decisions in changing weather conditions.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit pull requests to help improve the Rain Predictor App.

---

🌦️ **Predict Rain, Stay Dry, and Enjoy the Weather!** 🌈