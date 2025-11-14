# 🧬 DNA Sequence Classification – Coding vs Non-Coding  
*A Machine Learning + React + Flask Project*

This project classifies DNA sequences as **Coding** or **Non-Coding** using Machine Learning (Random Forest) and provides a clean **React UI** with a **Flask backend API**.

---

## 🚀 Features

### 🔬 Machine Learning  
- Uses **k-mer based vectorization (k=3)**  
- Trained on real DNA sequences labelled coding / non-coding  
- Random Forest model for sequence classification  
- Saves trained model as `dna_model.pkl` and `vectorizer.pkl`

### 🌐 Backend (Flask API)
- `/predict` endpoint accepts DNA sequence  
- Preprocesses → Vectorizes → Predicts  
- Includes CORS support  
- Easy to integrate with frontend / external systems

### 💻 Frontend (React)
- Clean & modern UI  
- Real-time DNA validation (only A, T, C, G allowed)  
- Sends prediction requests to Flask backend  
- Displays classification result instantly  

---

## 📂 Project Structure
Dna_Sequence_Classification/
│
├── backend/
│ ├── app.py
│ ├── train_model.py
│ ├── test_request.py
│ ├── test_model_direct.py
│ ├── dna_model.pkl
│ ├── vectorizer.pkl
│ └── requirements.txt
│
├── frontend/
│ ├── src/
│ │ ├── App.js
│ │ ├── DNAForm.js
│ │ └── index.js
│ ├── package.json
│ └── public/

## 🛠️ Tech Stack

### **Machine Learning**
- Python
- pandas
- scikit-learn
- joblib

### **Backend**
- Flask  
- Flask-CORS  

### **Frontend**
- React.js  
- Axios  
- CSS / JSX  

## 📌 How the Model Works

### 🧪 1. Preprocessing  
Each DNA sequence is converted into **3-mers**:  
ATCGA → ATC, TCG, CGA

### 🧩 2. Feature Extraction  
Using `CountVectorizer`, k-mers are transformed into a numerical vector.

### 🌲 3. Model  
A **Random Forest Classifier** learns patterns distinguishing:  
✔ Coding sequences  
✔ Non-coding sequences  

---

## ▶️ Running the Project

### **1. Start Backend**
```bash
cd backend
python app.py

Example Inputs
✔ Coding Sequence Example
ATGCGTACGTTAGCGCCGTACGCTAGC
Prediction: Coding

✔ Non-Coding Sequence Example
GCTTAGGCTAACCGATTAACCGGTTAG
Prediction: Non-Coding


