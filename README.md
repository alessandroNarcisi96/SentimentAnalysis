# Sentiment analysis

## Introduction

Sentiment analysis (or opinion mining) is a natural language processing (NLP) technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs

![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/sentiment.png)

This notebook aims at building a sentiment classifier from the content of the dataset that contains around 7000 short comments <br/>


## EDA

Is there any significant difference between a negative comment and a positive comment?<br/>
Let's start by plotting the wordclouds and the length distribuition of the comments

<p align="center">
    Negative Comments
</p>
<p align="center">
  <img alt="Light" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/negative_cloud1.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/dist_neg.png" width="45%">
</p>
<p align="center">
    Positive Comments
</p>
<p align="center">
  <img alt="Light" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/positive_cloud1.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/dist_pos.png" width="45%">
</p>


## MILESTONE 1:DATA CLEANING<br/>

The first challenge to deal with is that the comments are full of noise and useless expressions such as punctuation and emoticons that our models will not consider.<br/>
For example:” YTuber's watch [and if you can, comment] on @Jess' new vid, please! Her work *never* gets enough views/comments!”<br/>
A very common quote in ML comes to our minds “Garbage in garbage out”<br/>
In cleaner.py I wrote a very simple method clean() which is applied comment by comment and aims to remove all the words that start with ‘@’, all the references to websites and removes all the characters unless they are numbers or letters.<br/>
In addition to that,all the words are lowered.<br/>
Below the result:<br/>
“ytuber s watch  and if you can  comment  on jess  new vid  please  her work  never  gets enough views comments<br/>

## MILESTONE 2:TEXT EMBEDDING<br/>

A word embedding is a representation of a word in numerical format.This conversion is essential as models work with numbers.<br/>
Contextualizing word embeddings, as demonstrated by BERT, ELMo, and GPT-2, has proven to be a game-changing innovation in NLP.<br/> The use of contextualized word representations instead of static vectors (e.g., word2vec) has improved nearly every NLP task<br/>
Think about the word ’mouse’. It has multiple meanings, one of which refers to a rodent and the other to a device. Is BERT able to properly build one ’mouse’ representation per word sense?<br/>
Let's dive deeper in the process:<br/>
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/mouse.png)
### Tokenization
Tokenizer takes the input sentence and will decide to keep every word as a whole word, split it into sub words(with special representation of first sub-word and subsequent subwords — see ## symbol in the example above) or as a last resort decompose the word into individual characters. <br/>
Because of this, we can always represent a word as, at the very least, the collection of its individual characters.

After that,each segment is converted into an assigned number or id.<br/>

![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/token.png)


### BERT 
Next we need to convert our data to tensors(input format for the model) and call the BERT model.

#### Understanding the Output
hidden_states has four dimensions, in the following order:<br/>

The layer number (13 layers) : 13 because the first element is the input embeddings, the rest is the outputs of each of BERT’s 12 layers.<br/>
The batch number<br/>
The word / token number<br/>
The hidden unit / feature number (768 features)<br/>

In this project, I use bert from AutoModel that encode the 12 layers to get the final representation of the sentence composed by 768 features<br/>
The final outcome will be (batch X 768)<br/>

## MILESTONE 3:MODEL TUNING<br/>

### Hidden layer

In classifier.py I show the hidden layers which receive the result of bert and that will tune the model on the dataset<br/>
The three hidden layers are composed by 3 levels where each of them is followed by a Dropout.<br/>
The dropout layer offers a very computationally cheap and remarkably effective regularization method to reduce overfitting and improve generalization error in deep neural networks of all kinds.<br/>

Finally, the result is normalized by a softmax.<br/>

### Training

The loss function used is the CrossEntropyLoss with a batch size of 32 over 10 epochs.<br/>
In order to detect overfitting and ensure stability,cross-validation is applied.<br/>


As shown,the test line is quite fluctuating until the 6th epoch.After that,the accuracy results to be very stable and coherent with the training accuracy.<br/>


![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/tensorboard.png)

### Explainability
We can't rely on a model that we can't understand.<br/>
For this reason,in this project I've introduced a framework based on LIME that allows us to understand what the model considers relevant and how much.<br/>

What is LIME?<br/>
LIME ( Local Interpretable Model-agnostic Explanations )is a novel explanation technique that explains the prediction of any classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.<br/>

Below there are two very simple examples namely "I love you" and "I hate you".<br/>
<b>The result will be the role that every word has in the sentence and its height.</b><br/>
In this way it's possible to understand whether the model is truly understanding what the sentence actually means and it offers us a window on its way to think<br/>
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/exp.png)


### How can use explainability?<br/>
We can use this technique to check how model is learning and getting new knowledge over the epochs.<br/>
For example look at the following sentence:<br/>
<b>"i fucking hate my computer  it s all fucked up so now i can t listen to music  or do anything else much"</b>
<br/>
This sentence is clearly negative,but let'see how our model classifies it during the training:
<br/><br/>
&nbsp; &nbsp;

![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/fuck_1.png)
 &nbsp; &nbsp;
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/fuck_2.png)
 &nbsp; &nbsp;
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/fuck_4.png)
 &nbsp; &nbsp;
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/fuck_6.png)



## MILESTONE 4:MODEL RESULTS

### Results
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/report2.png)

### Understading the misclassified comments to improve the performance
A sentence clearly positive,but that our models classifies as a negative is the following:<br/>
<b>"only one exam left and i am so happy for it d"</b>
Let's dive deeper:<br/>
![alt text](https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/misclassify.png)
The model detect correctly that happy is a positive word,but it thinks that "only" and "exam" have a negative meaning.<br/>
Why could it be the case?<br/>
If we search into the training dataset we can see that there are 148 sentences that contain "exam" or "only", but just 50 are positive comments.<br/>
So the first problem to address is to ensure that we have enough examples to help the model to understand better the exact meaning of the vocabulary.<br/>
Is there a relation betweem the length of the sentence and the performance?<br/>
Let's plot the distribuition:<br/>



<p align="center">
  <img alt="Light" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/dist_len_1.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;

  <img alt="Dark" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/dist_len_pos_1.png" width="45%">
  </p>
&nbsp; &nbsp; &nbsp; &nbsp;
<p align="center">
  <img alt="Dark" src="https://github.com/alessandroNarcisi96/SentimentAnalysis/blob/master/Images/dist_len_neg_1.png" width="45%">
</p>