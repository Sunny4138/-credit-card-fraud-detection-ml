## 🛡️ Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent credit card transactions using advanced machine learning techniques. The system is designed to handle highly imbalanced real-world financial data where fraudulent activity is extremely rare compared to normal transactions.



## 🚀 Project Overview

• Developed an end-to-end fraud detection pipeline using Python & Machine Learning.

• Trained on the Kaggle Credit Card Fraud Dataset (284,807 transactions).

• Preprocessed data by removing noisy features, scaling transaction amounts, and applying SMOTE to handle heavy class imbalance.

• Built and optimized two core models:
  - Random Forest Classifier
  - XGBoost Classifier

• Used RandomizedSearchCV with StratifiedKFold for efficient hyperparameter tuning.

• Evaluated models using precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.

• Determined that XGBoost delivered the best overall fraud detection performance.

• Exported trained models using Joblib for deployment and real-time inference.



## 🛠️ Technologies Used

• Language: Python

• Libraries & Tools:
  - Pandas, NumPy
  - Scikit-Learn
  - Imbalanced-Learn (SMOTE, Pipeline)
  - XGBoost
  - Matplotlib, Seaborn
  - Joblib

• Hardware: CPU-based environment (No GPU required)

## 📁 Project Structure


```
project/
├── data/                     # Dataset files
│   └── creditcard.csv
├── models/                   # Saved ML models
│   ├── fraud_best_rf.pkl
│   ├── fraud_best_xgb.pkl
│   └── amount_scaler.pkl
├── notebooks/
│   └── fraud_detection.ipynb # Full Jupyter Notebook
├── src/
│   ├── train_model.py        # Model training script
│   ├── evaluate.py           # Evaluation logic
│   └── inference.py          # Prediction script for new data
├── screenshots/              # Plots, results, graphs
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```


## 🔧 Installation



-- git clone https://github.com/your-username/credit-card-fraud-detection.git
-- cd credit-card-fraud-detection
-- pip install -r requirements.txt



## 🧪 Training


-- python src/train_model.py


or open the Jupyter notebook:

-- jupyter notebook notebooks/fraud_detection.ipynb


## 📊 Evaluation

-- Key performance metrics:

-- Model	F1-Score	ROC-AUC	PR-AUC
-- XGBoost	0.78	0.99	0.85
-- RandomForest	0.68	0.98	0.82

-- XGBoost performed best, especially in recall and fraud-class detection.

## 📈 Results Summary

• Fraud detection improved significantly after applying SMOTE.

• XGBoost detected 103 true fraud cases with only 20 missed.

• Random Forest detected 101 true fraud cases but with slightly lower precision.

• Both models achieved near-perfect ROC-AUC scores (0.98–0.99).

• PR-AUC showed strong performance in imbalanced scenarios.




## 🔍 Inference (Predict Fraud)

-- python src/inference.py --model models/fraud_best_xgb.pkl --input transactions.csv


## 📦 License

-- This project is licensed for academic and educational use.



## ✍️ Author

-- If you need help with training, evaluation, or improving the machine learning model, feel free to reach out!

📩 Email: sunnyk36803@gmail.com

🔗 LinkedIn: www.linkedin.com/in/sunny30


