# Customer Churn Prediction Dashboard

This project is a **Customer Churn Prediction Dashboard** built using Python and Streamlit. The application allows users to predict the likelihood of customers churning from a bank based on their demographic and account information. The prediction is powered by several machine learning models, and users can interactively explore customer data, model probabilities, and explanations for predictions.

---

## Features

- **Interactive Dashboard**:
  - Input customer details or select from existing customer data.
  - View predictions and probabilities from multiple models.
  - Visualize average churn probabilities using gauge and bar charts.

- **Machine Learning Models**:
  - Includes XGBoost, Random Forest, K-Nearest Neighbors, and more.
  - Displays the contribution of models and feature importance.

- **Explainable AI**:
  - Generate detailed explanations for churn predictions.
  - Summarize model insights and customer-specific risk factors.

- **Customer Engagement**:
  - Create personalized emails with incentives to retain customers.

---

## Technologies Used

- **Frontend**:
  - [Streamlit](https://streamlit.io/) for building an interactive web-based dashboard.

- **Backend**:
  - Python for data processing and model handling.
  - `pickle` for loading pre-trained models.

- **Machine Learning**:
  - XGBoost, Random Forest, KNN, SVM, and more.

- **Visualization**:
  - Plotly for gauge and bar charts.

- **OpenAI API**:
  - Used for generating explanations and personalized emails.

### Machine Learning
The churn prediction model in this project is powered by various machine learning algorithms, which were pre-trained and saved as `.pkl` files. The key models used are:

- **XGBoost**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Decision Trees**
- **Voting Classifiers**: Combines the predictions of multiple models.

These models are used to predict customer churn probabilities based on input data.

### Model Inference
Inference is made by feeding customer information into the pre-trained models. The models output churn probabilities, which are then visualized in the dashboard and used for generating explanations and emails. The models' outputs are used to create a comprehensive overview of the customer's risk of churning.

- **Model Prediction**: The models return the probability of the customer churning.
- **Model Probabilities**: The dashboard visualizes the probabilities for each model and calculates the average probability of churn.

### Feature Engineering
Feature engineering plays a crucial role in preparing the customer data for the machine learning models. This includes transforming raw data into features that the models can interpret. Key steps include:

- **Encoding categorical variables**: Converting geographical location and gender into binary features (e.g., `Geography_France`, `Gender_Male`).
- **Handling numerical inputs**: Customer attributes like credit score, age, balance, and salary are passed as numerical features.
- **Handling missing data**: Ensuring there are no missing or invalid inputs before passing the data to models.

## Contributing

Contributions to the project are welcome! Feel free to submit bug reports, feature requests, or pull requests to help improve the application.

## License

This project is licensed under the [Apache License 2.0](/LICENSE).
