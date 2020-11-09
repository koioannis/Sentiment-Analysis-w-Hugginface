import sys, getopt
import spacy
import json
from spacy.lang.en import English
from pathvalidate import validate_filename
from model import SentimentAnalysis

def get_files():
    input_text = ''
    output_file = ''
    argv = sys.argv[1:]

    try:
        opts, _ = getopt.getopt(argv, "i:o:")    
    except:
        print('Wrong arguments')
        exit()

    for opt, arg in opts:
        try:
            validate_filename(arg)
        except:
            print("Enter valid filenames")
            exit()

        if (not (arg.endswith('.txt') or arg.endswith('.json'))):
            print('Wrong file extensions.\nInput file must contain the .txt extension\nOutput file must contain the .json extension')
            exit()

        if opt in ['-i']:
            with open(arg) as file:
                input_text = file.read()
        
        if opt in ['-o']:
            output_file = arg

    return input_text, output_file

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
    input_text, output_file = get_files()

    # split text into sentences
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    doc = nlp(input_text)
    input_text = tuple(span.text for span in doc.sents)

    print(f'Calculating Sentiment Metric for {len(input_text)} sentences...')

    # calculate sentiment metric
    sent = SentimentAnalysis(device='cpu')
    scores = sent.get_sentiment(input_text)

    # save json to output
    save_to_json(output_file, scores)
    print(f'Sentiment saved to {output_file}')



if __name__ == "__main__":
    main()