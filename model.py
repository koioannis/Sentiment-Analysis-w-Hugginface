import torch
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from util import sigmoid
from util import Singleton 

class SentimentModel(metaclass=Singleton):
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model.to(device if torch.cuda.is_available() else "cpu")
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
        try:
            encoded_review = self.tokenizer.encode_plus(
            sent,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            )
            input_ids = encoded_review['input_ids'][:512].to(self.device) 
            attention_mask = encoded_review['attention_mask'][:512].to(self.device)
            output = self.model(input_ids, attention_mask)
            output = F.softmax(output[0],dim=1).detach().cpu().numpy()[0]
            return output
        except:
            print('Cannot calculate sentiment for this sentence aborting..')
            return np.array([0, 0, 0, 0, 0])


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

        print(f'========================\nCalculating sentiment for:\n {sentences}\n')
        for sent in sentences:
            output = self.__predict(sent=sent)
            outputs += output

        outputs /= len(sentences)
        outputs = np.around(output, decimals=3)
        prob = self.__convert_five_star_system(outputs)
        
        return prob
    
    def __convert_five_star_system(self, outputs):
        sum = outputs[4] + outputs[3] + outputs[2] - outputs[1] - outputs[0]
        return sum if sum == 0 else sigmoid(sum)   
    