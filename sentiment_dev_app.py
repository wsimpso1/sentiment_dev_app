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

text = st.text_area('Enter Text to Analyze')
radio_selection = st.radio("Or select one of following sample sentences to analyze",
    ["The deployment of artifical intelligence (AI) on battlefields has provided life-saving opportunities to the war figher but also introduced new security vulnerabilities.",
    "Overall, the treaty was problematic for Vietnam, while strengthening France."],
    index=None
)
if radio_selection != None:
    text = radio_selection

run = st.button("Run", type="primary")

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

    st.subheader('Results (Sentence Level)')
    st.write('**Sentiment**:', sa_label)
    st.write('**Probability**:', sa_prob)
    
def run_ent_senti():
    with st.spinner('Loading Entity Model and Preprocessing Text'):
        absa = EntitySentimentTransformerOOB()
        output = absa.compute_sentiment(text, ent)
    success = st.success('Complete!')
    success.empty()

    st.subheader('Results (Entity Level)')
    st.write('**Chosen Entity**:', output[0])
    st.write('**Sentiment**:', output[1][0]['label'])
    st.write('**Probability**:', output[1][0]['score'])


if run:
    if sentence_level and absa_level:
        with col1:
            run_sent_senti()
        with col2:
            run_ent_senti()
    if sentence_level and not absa_level:
        run_sent_senti()

        # with st.spinner('Loading Sentence Model and Preprocessing Text'):
        #     sentsa = TransformerSentimentOOB()
        #     output = sentsa.compute_sentiment(text)
        # success = st.success('Complete!')
        # success.empty()

        # sa_label = sentsa.to_class_labels(output)
        # sa_prob = output[sa_label]
        # st.write('Sentence Level Results')
        # st.write('Sentiment:', sa_label)
        # st.write('Probability:', sa_prob)

    if absa_level and not sentence_level:
        run_ent_senti()

        # with st.spinner('Loading Entity Model and Preprocessing Text'):
        #     absa = EntitySentimentTransformerOOB()
        #     output = absa.compute_sentiment(text, ent)
        # success = st.success('Complete!')
        # success.empty()

        # st.write('Entity Level Results')
        # st.write('Chosen Entity:', output[0])
        # st.write('Sentiment:', output[1]['label'])
        # st.write('Probability:', output[1]['score'])
