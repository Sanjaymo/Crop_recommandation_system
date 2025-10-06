# 🌾 Crop Recommendation System

A machine learning-based crop recommendation system that predicts the most suitable crop to grow based on soil and environmental parameters using **XGBoost** classifier.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

This system helps farmers and agricultural professionals make informed decisions about crop selection by analyzing:
- Soil nutrients (Nitrogen, Phosphorus, Potassium)
- Environmental factors (Temperature, Humidity, Rainfall)
- Soil pH levels

The model provides crop recommendations with confidence scores and multiple alternatives.

## ✨ Features

- **High Accuracy**: XGBoost classifier with optimized hyperparameters
- **Multiple Recommendations**: Get top 3 crop suggestions with confidence scores
- **Interactive Input**: User-friendly command-line interface
- **Feature Importance**: Understand which factors matter most
- **Model Persistence**: Save and load trained models
- **Batch Prediction Support**: Use dictionary input for automated predictions

## 📊 Dataset

The `Crop_recommendation.csv` dataset should contain the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| N | Nitrogen content in soil | kg/ha |
| P | Phosphorus content in soil | kg/ha |
| K | Potassium content in soil | kg/ha |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH value | 0-14 |
| rainfall | Average rainfall | mm |
| label | Target crop name | - |

**Example crops**: Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**
```bash
cd CropRecommendation
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import xgboost; import sklearn; import pandas; print('All packages installed successfully!')"
```

## 💻 Usage

### Method 1: Interactive Menu

Run the main script:
```bash
python crop_model.py
```

You'll see a menu with options:
1. Train new model
2. Make prediction (interactive)
3. Make prediction (example data)
4. Exit

### Method 2: Direct Function Calls

#### Train the Model
```python
from crop_model import train_model

train_model()
```

#### Make Predictions (Interactive)
```python
from crop_model import predict_crop

# Interactive input
predict_crop()
```

#### Make Predictions (Programmatic)
```python
from crop_model import predict_crop

# Using dictionary input
crop_params = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.87,
    'humidity': 82.00,
    'ph': 6.50,
    'rainfall': 202.93
}

recommended_crop = predict_crop(crop_params)
```

### Example Output
```
============================================================
PREDICTION RESULTS
============================================================

🌾 Recommended Crop: RICE
📊 Confidence: 95.67%

------------------------------------------------------------
Top 3 Crop Recommendations:
------------------------------------------------------------
1. rice           : 95.67% ███████████████████
2. cotton         :  3.21% ▌
3. jute           :  1.12% ▏
============================================================
```

## 🔧 Model Details

### Algorithm: XGBoost Classifier

**Hyperparameters:**
- `n_estimators`: 200
- `max_depth`: 6
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `objective`: multi:softmax
- `eval_metric`: mlogloss

### Model Performance
- Achieves 95%+ accuracy on test data
- Stratified train-test split (80-20)
- Multi-class classification with balanced sampling

### Saved Files
After training, the following files are generated:
- `xgb_crop_model.pkl` - Trained XGBoost model
- `label_encoder.pkl` - Label encoder for crop names
- `feature_names.pkl` - Feature names for validation

## 📁 Project Structure

```
CropRecommendation/
├── Crop_recommendation.csv    # Dataset
├── crop_model.py              # Main script
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── .gitignore                 # Git ignore rules
├── xgb_crop_model.pkl        # Trained model (generated)
├── label_encoder.pkl         # Label encoder (generated)
└── feature_names.pkl         # Feature names (generated)
```

## 🛠️ Customization

### Adjusting Model Parameters

Edit the `train_model()` function in `crop_model.py`:

```python
clf = XGBClassifier(
    n_estimators=300,      # Increase for better accuracy
    max_depth=8,           # Increase for more complex patterns
    learning_rate=0.05,    # Decrease for more conservative learning
    # ... other parameters
)
```

### Adding New Features

1. Update your CSV with new columns
2. The model will automatically detect and use them
3. Retrain the model

## 📈 Tips for Best Results

1. **Data Quality**: Ensure your input data is accurate and within reasonable ranges
2. **Regular Updates**: Retrain the model periodically with new data
3. **Feature Scaling**: The model handles this automatically, but normalized data helps
4. **Multiple Predictions**: Check top 3 recommendations for alternatives

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Ideas for Enhancement
- Add web interface using Flask/Streamlit
- Implement ensemble models
- Add visualization of feature importance
- Include crop yield prediction
- Add regional crop databases

## 📝 License

This project is open-source and available for educational and research purposes.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Farming! 🌱**
