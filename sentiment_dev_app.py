import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class TransformerSentimentOOB:
    '''
    uses an OOB HuggingFace transformer model to extract sentiment from a sentence
    
    Attributes
    ----------
    model_path: str
        path to pretrained HF Transformer model 
    '''

    def __init__(self, model_path='finiteautomata/bertweet-base-sentiment-analysis'):
        self.pipeline = pipeline("text-classification", 
                                    model=model_path, 
                                    top_k=None)

    def to_class_labels(self, sentiment_score_dict):
        '''
        convert sentiment probability scores to single class label
        '''
        return max(sentiment_score_dict, key=sentiment_score_dict.get)

    def proportion_doc_sentiment(self,sent_list):
        doc_sentiment_list = self.compute_sentiment_sentence_list(sent_list)
        sent_labels = []
        for sent_sentiment_scores in doc_sentiment_list:
            label = self.to_class_labels(sent_sentiment_scores)
            sent_labels.append(label)
        count_of_sentiment = dict(Counter(sent_labels))
        doc_sentiment_proportion = {k: v / len(sent_list) for k, v in count_of_sentiment.items()}
        return doc_sentiment_proportion
    
    def compute_sentiment_sentence_list(self, sent_list):
        '''
        compute sentiment from list of sentences in a document

        '''
        return [self.compute_sentiment(sent) for sent in sent_list]

    def compute_sentiment(self, input):
        '''
        compute sentiment using pretrained transformer model

        Parameters:
        ------------
        input: str
            sentence from a document

        Returns:
        ---------
        sentiment_scores: dict
            computed sentiment scores (3 classes)
        '''
        # compute sentiment scores
        sentiment_scores = self.pipeline(input)[0]

        # convert list of dict to single dict
        new_sent_dict = {subdict['label']: subdict['score'] for subdict in sentiment_scores}
        
        return new_sent_dict
    
st.title('Sentiment Dev App')
st.subheader('Compute Sentiment of sentences', divider='green')
# st.markdown('User Authentication Required')

text = st.text_area('Enter Text', '')
run = st.button("Run", type="primary")

# create dataframe for preprocessing
inf_data = [{
    'id_num':0,
    'txt':text
    }]

all_data = []
for _ in inf_data:
    data = {'id':_['id_num'], 'text':_['txt']}
    all_data.append(data)

df = pd.DataFrame.from_dict(all_data)

if run:
    with st.spinner('Preprocessing Text'):
        sa = TransformerSentimentOOB()
        output = sa.compute_sentiment('This treaty introduced vulnerabilities for Vietnam')
    success = st.success('Complete!')
    success.empty()

    st.write(output)



    # sa = TransformerSentimentOOB()
    # output = sa.compute_sentiment('This treaty introduced vulnerabilities for Vietnam')
    # sa.compute_sentiment_sentence_list(sent_lst)
    # sa.proportion_doc_sentiment(sent_lst)