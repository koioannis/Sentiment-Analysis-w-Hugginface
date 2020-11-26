import sys, getopt
import spacy
import os
import json
from spacy.lang.en import English
from decouple import config

from model import SentimentModel
from data_handler import DataHandler



def get_filepaths():
    input_filepaths = []
    DATA_PATH = config('DATA_PATH')

    if (not os.path.isdir(DATA_PATH)):
        print("Make sure the directory that contains your csv exists (check .env file)")
        exit()

    for _, _, files in os.walk(DATA_PATH):
        for file in files:
            if (os.path.basename(file).endswith('.csv')):
                input_filepaths.append(os.path.abspath(f'{DATA_PATH }/{file}'))
    
    return input_filepaths

def main():
    nlp = English()
    filepaths = get_filepaths()

    sentiment_model = SentimentModel(device=config('CUDA_DEVICE'))
    data_handler = DataHandler(filepaths=filepaths,
        results_path=config('RESULTS_PATH'), 
        csv_sep=config('CSV_SEPERATOR'))
    sentencizer = nlp.create_pipe("sentencizer")

    nlp.add_pipe(sentencizer)
    texts_from_files = data_handler.get_text_from_files()
    scores_of_files = []
    for index, file_text in enumerate(texts_from_files):
        scores_of_files.append([])
        for text in file_text:
            # split text into sentences
            doc = nlp(text)
            text = tuple(span.text for span in doc.sents)
            scores_of_files[index].append(sentiment_model.get_sentiment(text))
        print(f'File {index+1}: DONE')
    
    data_handler.save_sentiment_to_files(scores_of_files)



if __name__ == "__main__":
    main()