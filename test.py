import streamlit as st
st.set_page_config(page_title="Sentiment Analyzer")
import pickle
path = ""

st.write("""
# Sentiment Prediction
## Submitted by: Abhishek Mamgain
### Selected Model:  Logistic Regression with Count Vectorizer
#
""")


filename = 'lr_model.sav'
lr_model = pickle.load(open(path+filename, 'rb'))

filename = 'cv.sav'
cv = pickle.load(open(path+filename, 'rb'))


text_inp = [st.text_input('Enter Text', "I love the pizza")]

tranformed_text = cv.transform(text_inp)

result = lr_model.predict(tranformed_text)
prob = lr_model.predict_proba(tranformed_text).max().round(4)
prob = (prob*100).round(2)
confidence_message = "Confidence: {}%".format(prob)

if st.button('Classify Text'):
    if result == 1:
        st.success("Sentiment: Positive")
        st.info(confidence_message)
    else:
        st.error("Sentiment: Negative")
        st.info(confidence_message)  

