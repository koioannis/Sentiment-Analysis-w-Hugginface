from model import sentAnal

def main():
    sent = sentAnal(max_seq_length=255, device='cpu')
    score = sent.get_sentiment(["This is a normal text"])
    print(score)

if __name__ == "__main__":
    main()