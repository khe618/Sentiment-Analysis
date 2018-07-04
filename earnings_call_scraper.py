import requests
import nltk
import csv
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score

def get_texts():
    """Retrieves the text from earning calls obtained from SeekingAlpha

    Returns:
    (list): An array of strings containing the text from each url
    
    """
    with open('urls.txt') as f:
        urls = f.read()
    urls = urls.split("\n")
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    texts = []
    for url in urls:
        html = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html, 'html.parser')
        soup.prettify()
        strong_tags = [paragraph.get_text() for paragraph in soup.find_all("strong")]
        first_speaker = strong_tags[strong_tags.index("Operator") + 2]
        if first_speaker == "Operator":
            first_speaker = strong_tags[strong_tags.index("Operator") + 3]
        elif first_speaker == "Question-and-Answer Session":
            first_speaker = strong_tags[strong_tags.index("Operator") + 1]
        text = [paragraph.get_text() for paragraph in soup.find_all("p")]
        text = "\n".join(text)
        text = text[text.index(first_speaker + "\n"):text.rindex("Question-and-Answer Session")]
        texts.append(text)
    return texts
        

def get_unigrams(texts):
    """Extracts the number of times a word occurs in the list of texts

    Args:
        texts (list): An array of strings containing the text from each url

    Returns:
        (dict): A dictionary containing each word as a key with the number of
        texts it occurs in as its value
        
    """
    unigrams = defaultdict(int)
    for text in texts:
        words = nltk.word_tokenize(text)
        for word in words:
            unigrams[word.lower()] += 1
    return unigrams

def get_unigram_features(unigrams, n):
    """Creates a list of n unigram features to train on

    Args:
        unigrams (dict): A dictionary containing each word as a key with the number of
        texts it occurs in as its value
        n (int): The number of features to be obtained. Must be less than len(unigrams)

    Returns:
        (list): A list containing the most common alphabetical unigrams

    """
    sorted_dict = sorted(unigrams, key=unigrams.get, reverse=True)
    unigram_features = []
    i = 0
    while len(unigram_features) < n:
        next_word = sorted_dict[i]
        if next_word.isalpha():
            unigram_features.append(next_word)
        i += 1
    return unigram_features

def get_bigrams(texts):
    """Extracts the number of times a pair of words occur in the list of texts

    Args:
        texts (list): An array of strings containing the text from each url

    Returns:
        (dict): A dictionary containing each pair of words as a key with the number
        of texts it occurs in as its value
        
    """
    all_bigrams = defaultdict(int)
    for text in texts:
        words = nltk.word_tokenize(text)
        for i in range(len(words) - 1):
            pair = (words[i].lower(), words[i+1].lower())
            all_bigrams[pair] += 1
    return all_bigrams

def get_bigram_features(bigrams, n):
    """Creates a list of n bigram features to train on

    Args:
        bigrams (dict): A dictionary containing each pair of words as a key with the number
        of texts it occurs in as its value
        n (int): The number of features to be obtained. Must be less than len(bigrams)

    Returns:
        (list): A list containing the most common alphabetical bigrams

    """
    sorted_bigrams = sorted(bigrams, key=bigrams.get, reverse = True)
    bigram_features = []
    i = 0
    while len(bigram_features) < n:
        next_bigram = sorted_bigrams[i]
        if next_bigram[0].isalpha() and next_bigram[1].isalpha():
            bigram_features.append(next_bigram)
        i += 1
    return bigram_features

def extract_features(text, unigram_features, bigram_features):
    """Converts a text into an array of features, specified by the variables unigram_features
    and bigram_features

    Args:
        text (str): A string containing an earnings call transcript
        unigram_features (list): A list containing the unigram_features to train on
        bigram_features (list): A list containing the bigram_features to train on

    Returns:
        (list) An array of ints corresponding to the number of times the text contains the
        specified features
        
    """
    words = nltk.word_tokenize(text)
    text_features = []
    for i in range(len(unigram_features)):
        text_features.append(words.count(unigram_features[i]))
    for i in range(len(bigram_features)):
        count = 0
        for j in range(len(words) - 1):
            if words[j] == bigram_features[i][0] and words[j+1] == bigram_features[i][1]:
                count += 1
        text_features.append(count)
    return text_features
    
    

with open("call_transcripts.txt") as f:
    texts = f.read()
texts = texts.split("SEPARATOR")

#texts = get_texts()
with open("quarterly_earnings_data.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    targets = list(reader)
targets = [int(x[1]) for x in targets]
unigram_features = get_unigram_features(get_unigrams(texts), 1000)
bigrams = get_bigrams(texts)
bigram_features = get_bigram_features(bigrams, 500)
text_features = [extract_features(text, unigram_features, bigram_features) for text in texts]
clf = svm.SVC(gamma=0.001, C=100.)
scores = cross_val_score(clf, text_features, targets, cv=5)

