import re
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer

# from nltk.corpus import stopwords

# Defining dictionary containing all emojis with their meaning

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

# Defining set containing all stopwords in english.
# stopwordlist = set(stopwords.words("english"))

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']


def preprocess(textdata):
    processedText = []

    # Lemmatizer and Stemmer
    wordlemm = WordNetLemmatizer()

    # Defining regex pattern.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()

        tweet = re.sub(urlPattern, ' URL', tweet)  # Using "URL" in place of URL's

        for emoji in emojis.keys():  # Replacing emojis
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

        tweet = re.sub(userPattern, ' USER', tweet)  # Replacing @username with USER

        tweet = re.sub(alphaPattern, " ", tweet)  # Replacing all non alphabets

        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)  # Replacing 3 more consecutive letters by 2 letters

        tweetwords = ''

        for word in tweet.split():

            if word not in stopwordlist:
                if len(word) > 1:
                    word = wordlemm.lemmatize(word)
                    tweetwords += (word + ' ')

        processedText.append(tweetwords)

    return processedText


def load_models():
    file = open('Data/vectoriser-ngram(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    file = open('Data/BNBmodel.pickle', 'rb')
    BNBmodel = pickle.load(file)
    file.close()

    return vectoriser, BNBmodel


def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)

    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ['Negative', 'Positive'])
    return df


if __name__ == '__main__':
    vectoriser, BNBmodel = load_models()

    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]

    df = predict(vectoriser, BNBmodel, text)
    print(df.head())
