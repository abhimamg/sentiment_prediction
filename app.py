import streamlit as st
st.set_page_config(page_title="Sentiment Analyzer")
import pickle
path = ""

#NLTK
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')


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

text_inp = st.text_input('Enter Text', "I love the pizza")

###########################
tokenizer = RegexpTokenizer("[a-zA-Z@]+") # We only want words in text as punctuation and numbers are not helpful
ss = SnowballStemmer("english")
stop = stopwords.words('english') 
excluding = ['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "no", ]
stop = [words for words in stop if words not in excluding]

def clean_up(sentence):
    sentence  = tokenizer.tokenize(sentence) # Conerting in regualr expression
    sentence = [ss.stem(w) for w in sentence if w not in stop  ]  # Stemming and removing stop words
    return " ".join(sentence) # returning the sentence in the form of a string

text_inp = clean_up(text_inp)
###############################   
    
tranformed_text = cv.transform([text_inp])

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

