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

## Contributing

Contributions to the project are welcome! Feel free to submit bug reports, feature requests, or pull requests to help improve the application.

## License

This project is licensed under the [Apache License 2.0](/LICENSE).
