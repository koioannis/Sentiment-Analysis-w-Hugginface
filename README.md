# Sentiment-Analysis-w-Hugginface
Using hugginface's pre-trained Sentiment Analysis model to predict snetiment metrics. For more information visit the model's oficial [page](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## Installation
Clone this repo:
```sh
git clone git@github.com:koioannis/Sentiment-Analysis-w-Hugginface.git
cd Sentiment-Analysis-w-Hugginface
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.
```bash
pip install -r requirements.txt
```
## Usage
Given a text as a txt file, you can predict the sentiment score and save the results to a json file using the command below.
```sh
python run_model.py -i <input_file.txt> -o <output_file.json>
```
## Demo
If we suppose that the input file contains only one sentence, "What a wonderful day", then the output json file should look like
```json
{
    "sentiment":{
        "1 star": "0.004",
        "2 star": "0.003",
        "3 star": "0.017",
        "4 star": "0.093",
        "5 star": "0.882"
    }
}
```
## License
[MIT](https://choosealicense.com/licenses/mit/)
