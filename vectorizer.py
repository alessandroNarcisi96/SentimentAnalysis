import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer

def vectorize(comment):
    bow = CountVectorizer(max_features=100, ngram_range=(1,1))
    X = bow.fit_transform(comment).toarray()
    df_td = pd.DataFrame(X, columns= bow.get_feature_names_out())
    return df_td


def bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    #create a vector with the dimension of the dictionary that contains all the words
    result_vector = np.zeros(dict_size)
    for word in text.split(' '): #split by space the comment to get all the words
        if word in words_to_index: # if present,update the counter
            result_vector[words_to_index[word]] +=1
    return result_vector
