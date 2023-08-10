from functools import reduce
import re
import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



def clean(comments):

    return comments.str.lower().replace("@\w+|[`!@#$%^&*()_+\-=\[\]{};':\"\|,.<>\/?~]|http.+? ", ' ',regex=True)


    # stop = stopwords.words('english')
    # text = " ".join([word for word in text.split() if word not in (stop)])
    # stemmer = PorterStemmer()
    # words = word_tokenize(text)

    # # using reduce to apply stemmer to each word and join them back into a string
    # stemmed_sentence = reduce(lambda x, y: x + " " + stemmer.stem(y), words, "")


    return comments