# 🎬 Sentiment Analysis of Movie Review

A simple and interactive web app built using **Streamlit** that classifies movie reviews as **positive** or **negative** using a pre-trained Logistic Regression model and TF-IDF vectorization.

## 🧠 How It Works

This app uses basic **Natural Language Processing (NLP)** techniques such as:
- Lowercasing
- Tokenization
- Removing stopwords
- Stemming using `PorterStemmer`

The preprocessed text is then vectorized using **TF-IDF** and classified using a **Logistic Regression** model trained on a labeled dataset of movie reviews.

---

## 📂 File Structure

```

.
├── app.py                         # Main Streamlit app
├── data/
│   ├── lr.pkl                    # Trained Logistic Regression model
│   └── tfidf.pkl                 # TF-IDF Vectorizer
└── README.md                     # Project documentation

````

---

## 🖥 Features

* Accepts user input movie review.
* Preprocesses and transforms the review text.
* Uses a trained model to predict sentiment.
* Provides feedback via Streamlit UI.

---

## 🔧 Model & Vectorizer

* **Model**: Logistic Regression
* **Vectorizer**: TF-IDF
* **Dataset**: Trained on a 50K IMDB Movie Reviews dataset (binary sentiment labels).

---
## 📊 Dataset

The model was trained using the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), a popular dataset containing 25,000 positive and 25,000 negative labeled reviews. This dataset is widely used for binary sentiment classification tasks.

---

## ✍️ Author

**Partho Sarothi Das**
*Aspiring Data Scientist | Passionate about ML & Visualization*
📧 Email: [partho52@gmail.com](mailto:partho52@gmail.com)

---

## 📜 License

This project is open-source and available for educational purposes.
