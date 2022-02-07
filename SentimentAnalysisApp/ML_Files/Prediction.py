import os

from .Text_Proprocessing import preprocess


def predict(text):
    import pickle
    print(os.getcwd())
    file = open('SentimentAnalysisApp/ML_Files/vectoriser-ngram(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    file = open('SentimentAnalysisApp/ML_Files/BNBmodel.pickle', 'rb')
    model = pickle.load(file)
    file.close()

    text = [text]
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)

    return sentiment
