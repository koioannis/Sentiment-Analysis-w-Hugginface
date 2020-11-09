import torch
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalysis:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.device = device

    def __predict(self, sent):
        """predicts the sentiment score for a single sentence
        
        Parameters:
        sent : str
            the sentence
        
        Return:
        output: ndarray
            1D array containing the sentiment score for the sentence
        """

        encoded_review = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
            
        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        output = self.model(input_ids, attention_mask)
        output = F.softmax(output[0],dim=1).detach().numpy()[0]
        return output

    def get_sentiment(self, sentences):
        """Calculates the mean sentiment score of all sentences
        
        Parameters:
        sentences : tuple
            A tuple containing the sentences
        
        Return:
        output: tuple
            A tuple containing the mean sentiment score of all sentences
        """

        outputs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        for sent in sentences:
            output = self.__predict(sent=sent)
            outputs += output

        outputs /= len(sentences)
        outputs = np.around(output, decimals=3)
        return tuple(outputs)