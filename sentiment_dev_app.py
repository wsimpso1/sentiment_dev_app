import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import warnings
from sentiment import TransformerSentimentOOB, EntitySentimentTransformerOOB
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# # start pre loading models
# sentsa = TransformerSentimentOOB()
# absa = EntitySentimentTransformerOOB()

    
st.title('Sentiment Dev App')
st.subheader('Compute Sentiment of sentences', divider='green')
# st.markdown('User Authentication Required')

text = st.text_area('Enter Text', key='The deployment of artifical intelligence (AI) on battlefields has provided life-saving opportunities to the war figher but also introduced new security vulnerabilities')
run = st.button("Run", type="primary")

col1, col2 = st.columns(2)
with col1:
    sentence_level = st.checkbox("Sentence Level Sentiment")
with col2:
    absa_level = st.checkbox("Entity Level Sentiment")
if absa_level:
    ent = st.text_input('Enter an entity to evaluate')

if run:
    if sentence_level:
        with st.spinner('Loading Sentence Model and Preprocessing Text'):
            sentsa = TransformerSentimentOOB()
            output = sentsa.compute_sentiment(text)
        success = st.success('Complete!')
        success.empty()

        st.write(output)

    if absa_level:

        with st.spinner('Loading Entity Model and Preprocessing Text'):
            absa = EntitySentimentTransformerOOB()
            output = absa.compute_sentiment(text, ent)
        success = st.success('Complete!')
        success.empty()

        st.write(output)

