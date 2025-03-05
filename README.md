# 🎵 Music Genre Classification using Machine Learning

## 📌 Project Overview
This project is a **music genre classification system** that uses **multiple machine learning models** to predict the genre of a given `.wav` audio file. It extracts **MFCC features** from the audio and trains various models to classify the genre. The user can upload a song, and the best-performing model will predict the genre.

## 🏗️ Project Structure
```
music_genre_classification/
│── main.py                  # Main Streamlit UI
│── dataset.py                # Dataset generation and loading
│── models.py                 # Model training and prediction
│── utils.py                  # Utility functions (e.g., feature extraction)
│── requirements.txt          # Dependencies
│── README.md                 # Project Documentation
```

## 🔧 Installation and Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Padmanabhannm18/music-genre-classification.git
cd music-genre-classification
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## 🚀 Running the Application

After installing dependencies, run the Streamlit app:
```sh
streamlit run main.py
```

## 📂 File Descriptions

### `main.py`
- Launches the **Streamlit UI**.
- Uploads audio files for prediction.
- Displays model performance and predicted genre.

### `dataset.py`
- **Generates the dataset** if it doesn't exist.
- **Loads and splits the dataset** into training and test sets.

### `models.py`
- **Trains multiple models**, including:
  - KNN
  - Random Forest
  - SVM
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Naive Bayes
  - MLP (Neural Network)
- Predicts the **genre** of an uploaded `.wav` file.
- Finds the **best model** based on accuracy.

### `utils.py`
- Extracts **MFCC features** from audio files for model training and prediction.

### `requirements.txt`
- Lists all dependencies required for the project.

## 🎯 Features
✅ Supports **multiple machine learning models** for classification.  
✅ Uses **MFCC features** to analyze audio signals.  
✅ **Trains models** and evaluates their accuracy.  
✅ **Uploads audio files** and predicts their genre.  
✅ Displays **model performance** in an interactive UI.  

## 📊 Dataset
The dataset consists of **10 music genres** from the **GTZAN dataset**:
- 🎸 **Blues**
- 🎼 **Classical**
- 🎶 **Country**
- 🕺 **Disco**
- 🎤 **Hiphop**
- 🎷 **Jazz**
- 🎸 **Metal**
- 🎵 **Pop**
- 🎹 **Reggae**
- 🤘 **Rock**

## 📌 Example Output
- **Best Model:** Random Forest (Accuracy: 85%)
- **Predicted Genre:** Rock 🎸

## 👨‍💻 Contributing
Feel free to contribute by improving the model performance or optimizing feature extraction!
1. Fork the repo
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Added new feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

## 🛠️ Technologies Used
- **Python** 🐍
- **Streamlit** 📊
- **Scikit-learn** 🤖
- **MFCC Feature Extraction** 🎼
- **NumPy & Pandas** 📊

## 📜 License
This project is open-source and available under the **MIT License**.

---

🎶 **Happy Coding!** 🚀

