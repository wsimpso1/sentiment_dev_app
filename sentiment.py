from collections import Counter
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    
class EntitySentimentTransformerOOB:

    def __init__(self, model_path='yangheng/deberta-v3-large-absa-v1.1'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def compute_sentiment(self, input, entity):
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
        sentiment_score = (entity, self.classifier(input, text_pair=entity))

        return sentiment_score
