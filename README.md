# 📊 Customer Churn Prediction using ANN  

## 📌 Overview  
This project predicts **customer churn** (whether a customer will leave or stay) using an **Artificial Neural Network (ANN)**.  
Churn prediction is widely used in telecom, banking, insurance, and SaaS companies to **retain valuable customers**.  

---

## ⚙️ Tech Stack  
- **Language:** Python  
- **Libraries:** TensorFlow / Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook / VS Code, GitHub  

---

## 🗂️ Dataset  
The dataset contains customer information such as:  
- **Demographics:** Age, Gender, Geography  
- **Account Info:** Balance, Salary, Tenure  
- **Activity:** Credit Score, Number of Products, Has Credit Card, Active Member  
- **Target:** Churn (1 = left, 0 = stayed)  

---

## 🧠 Model Architecture  
- Input layer → encoded features  
- Hidden layers → Dense + ReLU + Dropout  
- Output layer → Sigmoid activation (binary classification)  
- Loss → Binary Crossentropy  
- Optimizer → Adam  

---

## 📈 Results & Evaluation  
The model is evaluated using:  
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC-AUC  

---

## 🚀 How to Run  

1. **Clone the repo:**  
   ```bash
   git clone https://github.com/yourusername/CHURN-ANN-MODEL.git
   cd CHURN-ANN-MODEL
   ```  

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Run the model:**  
   ```bash
   python app.py
   ```  
   Or open the Jupyter notebook:  
   ```bash
   jupyter notebook
   ```  

---


## 📌 Future Improvements  
- Hyperparameter tuning (Grid/Random Search)  
- Handle class imbalance (SMOTE, class weights)  
- Model explainability with SHAP / LIME  
- Deploy with Streamlit Cloud / Heroku  

---

## 👨‍💻 Author  
**Shubhank Manhas**  
AI & ML Enthusiast | Data Science Learner  
