# Workplace Mental Health Predictor 🌿🧠
A machine learning web application built with Streamlit that analyzes workplace environment attributes and predicts potential mental health risks using Random Forest and XGBoost algorithms.

## 🚀 Features

- **Interactive Dashboard**: Clean and intuitive UI built with Streamlit
- **Two ML Models**:
  - Random Forest Classifier
  - XGBoost Classifier
- **Multi-class Classification**: Predicts one of three mental health risk levels:
  - 0: No risk
  - 1: Maybe
  - 2: Yes
- **Feature Importance**: Visualizes the most important factors influencing predictions
- **Real-time Prediction**: Get instant predictions based on your inputs
- **Data Preprocessing**: Automated handling of categorical features and encoding

## 📋 Prerequisites

- Python 3.8+
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Subhadip-Paul2006/workplace-mental-health-predictor-Parvati
   cd workplace-mental-health-predictor-Parvati
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### Train the Model

To train both Random Forest and XGBoost models:

```bash
python src/train.py
```

This will:
1. Load and preprocess the dataset
2. Train both models
3. Save them to the `models/` directory
4. Generate feature importance plots

### Run the Web Application

To start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser.

## 📊 Data

The dataset `data/raw/survey.csv` contains anonymous responses from employees about:
- Company size
- Tech company status
- Mental health benefits
- Care options availability
- Help-seeking behavior
- Wellness program presence
- Anonymity provisions
- Leave policy
- Perceived seriousness of mental health
- Comparison with physical health

## ⚙️ Configuration

See `src/config.py` for configuration options, including file paths and model settings.

## 📂 Project Structure

```
workplace-mental-health-predictor-Parvati/
├── app.py                  # Streamlit application
├── data/
│   └── raw/
│       └── survey.csv      # Original survey dataset
├── models/                 # Trained model files
│   ├── mental_health_rf_model.pkl
│   ├── mental_health_xgb_model.pkl
│   └── label_encoder.pkl
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model training
│   └── config.py           # Configuration
├── .gitignore              # Git ignore settings
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
