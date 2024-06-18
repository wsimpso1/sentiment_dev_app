import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import warnings
from sentiment import TransformerSentimentOOB
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    
st.title('Sentiment Dev App')
st.subheader('Compute Sentiment of sentences', divider='green')
# st.markdown('User Authentication Required')

text = st.text_area('Enter Text', '')
run = st.button("Run", type="primary")

# # create dataframe for preprocessing
# inf_data = [{
#     'id_num':0,
#     'txt':text
#     }]

# all_data = []
# for _ in inf_data:
#     data = {'id':_['id_num'], 'text':_['txt']}
#     all_data.append(data)

# df = pd.DataFrame.from_dict(all_data)

if run:
    with st.spinner('Preprocessing Text'):
        sa = TransformerSentimentOOB()
        output = sa.compute_sentiment(text)
    success = st.success('Complete!')
    success.empty()

    st.write(output)



    # sa = TransformerSentimentOOB()
    # output = sa.compute_sentiment('This treaty introduced vulnerabilities for Vietnam')
    # sa.compute_sentiment_sentence_list(sent_lst)
    # sa.proportion_doc_sentiment(sent_lst)

