from django.shortcuts import render
from .ML_Files.Prediction import predict
import nltk

dwlr = nltk.downloader.Downloader()
for pkg in dwlr.packages():
    if pkg.subdir == 'taggers':
        dwlr.download(pkg.id)

# Create your views here.

def home(request):
    return render(request, 'index.html')


def result(request):
    text = (str(request.GET['text']))
    # print(text)
    prediction = predict(text)
    print(prediction)
    result = ''
    if prediction == 1:
        result = 'Positive'
    else:
        result = 'Negative'

    return render(request, 'result.html', {'result': result})
