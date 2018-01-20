from __future__ import division
from nltk import RegexpTokenizer
import glob
import os
import sys
from stop_words import get_stop_words
import math

#Authors: Gokul Surendra , Padma Kurdgi*******************************************************

stop_words = get_stop_words('english')
tokenizer = RegexpTokenizer(r'\w+')
train_path = sys.argv[1]+"\\*"
test_path = sys.argv[2]+"\\*"
train_folders_list = glob.glob(train_path)
test_folders_list = glob.glob(test_path)


def calc_prior(folders_list):
    prior_count = dict()
    prior_prob = dict()
    total_count =0
    for folder in folders_list[:5]:
        target_class = folder.split("\\")[-1]
        file_count = len(os.listdir(folder))
        prior_count[target_class] = file_count
        total_count += file_count
    for folder in folders_list[:5]:
        target_class = folder.split("\\")[-1]
        prior_prob[target_class] = prior_count.get(target_class) / total_count
    return prior_prob

class Word():
    def __init__(self, word, prob, target_class):
        self.word = word
        self.prob = prob
        self.target  = target_class

def get_all_classes(folders_list):
    classes = []
    for folder in folders_list[:5]:
        target_class = folder.split("\\")[-1]
        classes.append(target_class)
    return classes


def extract_vocab(folder_list):
    vocab = []
    for folder in folder_list:
        file_list = os.listdir(folder)
        for file in file_list:
            words = extract_token_docs(folder, file)
            vocab.extend(words)
    return vocab

def concat_text_class(folders_list, target):
    words_in_class = []
    for folder in folders_list[:5]:
        target_class = folder.split("\\")[-1]
        if target_class == target:
            file_list = os.listdir(folder)
            for file in file_list:
                tokens = extract_token_docs(folder, file)
                words_in_class.extend(tokens)
    return words_in_class


def count_tokens(term, words_list):
    count =0
    for word in words_list:
        if word == term:
            count +=1
    return count


def train_multinomial_nb(folders_list):
    vocab = extract_vocab(folders_list)
    classes_list = get_all_classes(folders_list)
    unique_words = set(vocab)
    count_class_tokens = dict()
    text_class = dict()
    cond_prob = dict()
    for target_class in classes_list:
        text_class[target_class] = concat_text_class(folders_list, target_class)
        all_count = len(text_class[target_class])
        for term in unique_words:
            count_term = text_class[target_class].count(term)#count_tokens(term, text_class[target_class])
        #for term in unique_words:
            prob = (count_term+1)/(all_count+len(unique_words))
            word = Word(term, target_class, prob)
            cond_prob[term, target_class] = prob
    return cond_prob


def extract_token_docs(folder, file):
    tokens_list = []
    end_header = False
    with open(folder + '\\' + file, "r") as text_file:
        for line in text_file:
            if 'Lines:' in line:  # Ignore header ---  Till 'Lines:'
                end_header = True
            elif end_header:
                tokens = tokenizer.tokenize(line.lower())
                # Tokenize data
                for word in tokens:
                    if word.isalpha():
                        if word not in stop_words:
                            tokens_list.append(word)
    return tokens_list


def apply_multinomial_nb(cond_prob, doc_tokens, classes_list, prior_prob_class):
    score = {}
    for target in classes_list:
        score[target] = math.log10(prior_prob_class.get(target))
        for term in doc_tokens:
            if (term, target) in cond_prob.keys():
                score[target] += math.log10(cond_prob[term, target])
    return  max(score, key=lambda k: score[k])


def actual_classify(folders_list):
    actual = {}
    for folder in folders_list[:5]:
        target_class = folder.split("\\")[-1]
        file_list = os.listdir(folder)
        for file in file_list:
            actual[file] = target_class
    return actual

def process_test_data(folders_list, cond_probability, prior_prob_class):
    class_list = get_all_classes(folders_list)
    prediction = {}
    for folder in folders_list[:5]:
        file_list = os.listdir(folder)
        for file in file_list:
            tokens = extract_token_docs(folder, file)
            prediction[file] = apply_multinomial_nb(cond_probability, tokens, class_list, prior_prob_class)
    return prediction


def get_accuracy(classify, actual_classify):
    count = 0
    for entry, actual_entry in zip(classify.iteritems(), actual_classify.iteritems()):

        if entry == actual_entry:
            count += 1
    return (count/len(actual_classify))*100


def main():

    prior_prob_class = calc_prior(train_folders_list)
    cond_probability = train_multinomial_nb(train_folders_list)
    classify = process_test_data(test_folders_list, cond_probability, prior_prob_class)
    actual = actual_classify(test_folders_list)
    print 'Accuracy: '
    print get_accuracy(classify, actual)


if __name__ == '__main__':
    main()