# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:42:32 2021

@author: Qalbe
"""

#import the libraries

import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import matplotlib.pyplot as plt




#reading file
file = pd.read_csv("hiringg.csv")


# replacing nan with zero in case of strings
file.experience = file.experience.fillna("zero")


#converting words to numberss
file.experience = file.experience.apply(w2n.word_to_num)


# finding mean
import math
median_test_score = math.floor(file['test_score(out of 10)'].mean())
median_test_score


file['test_score(out of 10)'] = file['test_score(out of 10)'].fillna(median_test_score)


#Linear Regression
reg = linear_model.LinearRegression()
reg.fit(file[['experience','test_score(out of 10)','interview_score(out of 10)']],
              file['salary($)'])



reg.predict([[6,5,0]])


reg.predict([[15,13,10]])