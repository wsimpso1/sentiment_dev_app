import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import warnings
from sentiment import TransformerSentimentOOB, EntitySentimentTransformerOOB
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    
st.title('Sentiment Dev App')
st.subheader('Compute Sentiment of sentences', divider='green')
# st.markdown('User Authentication Required')

text = st.text_area('Enter Text', '')
run = st.button("Run", type="primary")


sentence_level = st.checkbox("Sentence Level Sentiment")
absa_level = st.checkbox("Entity Level Sentiment")
if absa_level:
    ent = st.text_input('Enter an entity to evaluate')

if run:
    if sentence_level:
        with st.spinner('Preprocessing Text'):
            sa = TransformerSentimentOOB()
            output = sa.compute_sentiment(text)
        success = st.success('Complete!')
        success.empty()

        st.write(output)

    if absa_level:

        with st.spinner('Preprocessing Text'):
            sa = EntitySentimentTransformerOOB()
            output = sa.compute_sentiment(text, ent)
        success = st.success('Complete!')
        success.empty()

        st.write(output)

