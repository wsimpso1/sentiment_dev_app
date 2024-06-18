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
st.subheader('Compute the Sentiment of Sentences', divider='green')
# st.markdown('User Authentication Required')

text = st.text_area('Enter Text to Analyze')
radio_selection = st.radio("Or select one of following sample sentences to analyze",
    ["Artificial Intelligence (AI) on the battlefield is a security vulnerability even if the war fighter stands to benefit.",
     "This feature of the weapons guidance mechanism reduces the effectiveness of the entire system.",
    "The Ambassador criticized the United Kingdom's handling of the situtation, although the outcome was ultimately advantageous."],
    index=None
)
if radio_selection != None:
    text = radio_selection

run = st.button("Run", type="primary")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        sentence_level = st.checkbox("Sentence Level Sentiment")
    with col2:
        absa_level = st.checkbox("Entity Level Sentiment")
        if absa_level:
            ent = st.text_input('Enter a relevant entity to evaluate')

def run_sent_senti():
    with st.spinner('Loading Sentence Model and Preprocessing Text'):
        sentsa = TransformerSentimentOOB()
        output = sentsa.compute_sentiment(text)
    success = st.success('Complete!')
    success.empty()

    sa_label = sentsa.to_class_labels(output)
    sa_prob = output[sa_label]
    
    if sa_label == 'NEU':
        sa_label = 'Neutral'
    elif sa_label == 'NEG':  
        sa_label = 'Negative'
    elif sa_label == 'POS':
        sa_label = 'Positive'

    st.subheader('Sentence Level Results')
    st.write('**Sentiment**:', sa_label)
    st.write('**Probability**:', sa_prob)
    
def run_ent_senti():
    with st.spinner('Loading Entity Model and Preprocessing Text'):
        if ent == '':
            st.write('No entity found. Please enter an entity.')
            return None
        absa = EntitySentimentTransformerOOB()
        output = absa.compute_sentiment(text, ent)
    success = st.success('Complete!')
    success.empty()

    st.subheader('Entity Level Results')
    st.write('**Chosen Entity**:', output[0])
    st.write('**Sentiment**:', output[1][0]['label'])
    st.write('**Probability**:', output[1][0]['score'])


if run:
    with st.container(border=True):
        cola, colb = st.columns(2)

        if sentence_level and absa_level:
            with cola:
                run_sent_senti()
            with colb:
                run_ent_senti()

        if sentence_level and not absa_level:
            run_sent_senti()

        if absa_level and not sentence_level:
            run_ent_senti()

