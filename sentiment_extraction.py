# -*- coding: utf-8 -*-
"""
This file takes the input from "formatted_tweets.txt" (output of format_tweets.py) and 
creates the file "semantic_metrics.txt" which contains the sentiment metrics of all tweets.
"""
from nltk.sentiment import SentimentAnalyzer
from nltk.corpus import subjectivity
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

filepath = 'formatted_tweets.txt'  #input file
with open('semantic_metrics.txt', 'w') as the_file: #output file
    with open(filepath) as fp:  
        line = fp.readline()
        count = 1
        sent_int_analyzer = SentimentIntensityAnalyzer()
        while line:
            print("Line {}: {}".format(count, line.strip()))
            scores = sent_int_analyzer.polarity_scores(line)
            #for item in ss:
                #the_file.write("%d," % ss[item])
            #the_file.write(ss)
            for value in scores.values():
                the_file.write('{},'.format(value))
            the_file.write('\n')
            print(scores)
            line = fp.readline()
            count += 1