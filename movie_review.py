import streamlit as st
import pickle
import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
# Logical Function
#---------------------------
def text_transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    alnum = []
    for i in text:
        if i.isalnum():
            alnum.append(i)
    alnum_stop = []
    for i in alnum:
        if i not in stopwords.words('english'):
            alnum_stop.append(i)
    final = []
    for i in alnum_stop:
        final.append(ps.stem(i))
    return " ".join(final)

# File Import
#------------------------
@st.cache_resource
def load_model():
    with open('data/lr.pkl', 'rb') as file:
        lr = pickle.load(file)
    return lr

@ st.cache_resource
def load_tfidf():
    with open('data/tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    return tfidf
lr = load_model()
tfidf = load_tfidf()

# Main body
# --------------------------
st.title('Sentiment analysis of movie review:')
review = st.text_area('Input a review:')
clicked = st.button('✅ Submit')
if clicked:
    if review =='':
        st.error('Please input a review to analyze.')
    else:
        transformed_review = text_transformer(review)
        review_vectorize = tfidf.transform([transformed_review]).toarray()
        pred = lr.predict(review_vectorize)
        if pred == 1:
            st.info('This is a positive review.')
        else:
            st.error('Sorry! This is a negative review.')


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; font-size: 0.9em; color: gray;">
    Created by <b>Partho Sarothi Das</b><br>
    <i>Aspiring Data Scientist | Passionate about ML & Visualization</i><br>
    Email: <a href="mailto:partho52@gmail.com">partho52@gmail.com</a>
</div>
""", unsafe_allow_html=True)