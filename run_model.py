import sys, getopt
import spacy
import os
import json
from spacy.lang.en import English

from model import SentimentModel
from data_handler import DataHandler



def get_filepaths():
    DATA_PATH = './data'
    input_filepaths = []

    if (not os.path.isdir(DATA_PATH)):
        print("Make sure data directory exists")
        exit()

    for _, _, files in os.walk(DATA_PATH):
        for file in files:
            if (os.path.basename(file).endswith('.csv')):
                input_filepaths.append(os.path.abspath(f'{DATA_PATH }/{file}'))
    
    return input_filepaths
    

def save_to_json(output_file, scores):
    json_data = {
        "sentiment":{
            "1 star": str(scores[0]),
            "2 star": str(scores[1]),
            "3 star": str(scores[2]),
            "4 star": str(scores[3]),
            "5 star": str(scores[4])
        }
    }

    with open(output_file, "w+") as json_file:
        json.dump(json_data, json_file)


def main():
    nlp = English()
    filepaths = get_filepaths()

    sentimentModel = SentimentModel(device='cpu')
    dataHandler = DataHandler(filepaths=filepaths)
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    files_text = dataHandler.get_text_from_files()
    files_scores = []
    for index, file_text in enumerate(files_text):
        files_scores.append([])

        print(f'Processing File {index+1}\n')
        for text in file_text:
            # split text into sentences
            doc = nlp(text)
            text = tuple(span.text for span in doc.sents)

            print(f'Calculating Sentiment Metric for {len(text)} sentences...')
            files_scores[index].append(sentimentModel.get_sentiment(text))
        print(f'File {index+1}: DONE')
    
    for files_score in files_scores:
        print(files_score)



if __name__ == "__main__":
    main()