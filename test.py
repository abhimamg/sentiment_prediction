import streamlit as st
st.set_page_config(page_title="Sentiment Analyzer")
import pickle
path = ""

st.write("""
# Assessemnt 3: Sentiment Prediction
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
confidence_message = "Confidence = {}%".format(prob)


# st.write(result)
# st.write(prob)

if st.button('Classify Text'):
    if result == 1:
        st.success("Sentiment: Positive")
        st.info(confidence_message)
    else:
        st.error("Sentiment: Negative")
        st.info(confidence_message)  



# st.write("""
# # Assessemnt 3: Sentiment Prediction
# ### Submitted by: Abhishek Mamgain
# #
# """)

# text_inp = st.text_input(
#     'Enter Text', "I really like this food")

# score = vader.polarity_scores(text_inp)

# comp_score = score["compound"]
# pos_score = round(score["pos"]*100,2)
# neg_score = round(score["neg"]*100,2)
# neu_score = round(score["neu"]*100,2)
# pos_message = "Confidence = {}%".format(pos_score)
# neg_message = "Confidence = {}%".format(neg_score)
# neu_message = "Confidence = {}%".format(neu_score)


# if st.button('Classify Text'):
#     if pos_score > 0.4:
#         st.success("Sentiment: Positive")
#         st.info(pos_message)
#     elif neg_score > 0.4:
#         st.error("Sentiment = Negative")
#         st.info(neg_message)     
#     else:
#         st.warning("Sentiment = Neutral")
#         st.info(neu_message)  