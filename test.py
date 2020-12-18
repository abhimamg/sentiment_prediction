import streamlit as st
st.set_page_config(page_title="Sentiment Analyzer")

import pandas as pd
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

st.write("""
# Assessemnt 3: Sentiment Prediction
### Submitted by: Abhishek Mamgain
#
""")


text_inp = st.text_input(
    'Enter Text', "I really like this food")

score = vader.polarity_scores(text_inp)

comp_score = score["compound"]
pos_score = round(score["pos"]*100,2)
neg_score = round(score["neg"]*100,2)
neu_score = round(score["neu"]*100,2)
pos_message = "Confidence = {}%".format(pos_score)
neg_message = "Confidence = {}%".format(neg_score)
neu_message = "Confidence = {}%".format(neu_score)


if st.button('Classify Text'):
    if pos_score > 0.4:
        st.success("Sentiment: Positive")
        st.info(pos_message)
    elif neg_score > 0.4:
        st.error("Sentiment = Negative")
        st.info(neg_message)     
    else:
        st.warning("Sentiment = Neutral")
        st.info(neu_message)  