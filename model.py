import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class sentAnal:
    def __init__(self, max_seq_length, device):
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.max_seq_length = max_seq_length
        self.device = device

    def __predict(self, sent):
        encoded_review = self.tokenizer.encode_plus(
            sent,
            max_length=self.max_seq_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
            
        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        output = self.model(input_ids, attention_mask)
        output = F.softmax(output[0],dim=1)

        return output

    def get_sentiment(self, sentences):
        probs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        for sent in sentences: 
            output = self.__predict(sent=sent)
            probs += output.detach().numpy()[0]

        probs /= len(sentences)
        return probs

if __name__ == "__main__":
    sentiment = sentAnal(max_seq_length=255, device='cpu')
    print(sentiment.get_sentiment(["what a great day"]))
    print(sentiment.get_sentiment(["what a bad day"]))