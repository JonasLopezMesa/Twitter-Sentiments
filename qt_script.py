#importar librerías
#LIBRERÍAS PARA QT
import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QBoxLayout, QPushButton
from PySide2.QtCore import Qt
###########################################
import os #para poder utilizar comandos de la terminal de windows
import os.path as path
import pandas as pd
from bs4 import BeautifulSoup
import requests
from requests_oauthlib import OAuth1
import string
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict
#para los otros tipos de algoritmos:
from sklearn.svm import SVC #Support Vector Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#########################################
#Para importar y exportar el algoritmo entrenado
from pickle import dump
from pickle import load
#########################################
import csv
import collections.abc
import collections
import re
from collections import Counter
import numpy as np
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')
#Datos para el entrenamiento
if  path.exists('fvecto'):
    fvecto = open('fvecto', 'rb') #abre el archivo en modo lectura
    vectorizer = load(fvecto) #carga el archivo en la variable 
else:
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
best_model = 9

class Ventana(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.label = QLabel("HOla")

        self.buttonBloqueFile = QPushButton("&Analizar Bloque desde archivo", self)
        self.buttonBloqueTwitter = QPushButton("&Analizar Bloque desde Twitter.com", self)
        self.buttonUnTweet = QPushButton("&Analizar un Tweet", self)
        self.buttonEntrenar = QPushButton("&Entrenar algoritmos", self)
        
        self.layoutMenu =QHBoxLayout()
        self.layoutMenu.addWidget(self.buttonBloqueFile)
        self.layoutMenu.addWidget(self.buttonBloqueTwitter)
        self.layoutMenu.addWidget(self.buttonUnTweet)
        self.layoutMenu.addWidget(self.buttonEntrenar)

        self.layoutWidget =QHBoxLayout()

        self.layoutPrincipal = QVBoxLayout()
        self.layoutPrincipal.addLayout(self.layoutMenu,2000)
        self.layoutPrincipal.addLayout(self.layoutWidget)


        self.setLayout(self.layoutPrincipal)

app = QtWidgets.QApplication(sys.argv) #variable para la interfaz
ventana = Ventana()
ventana.resize(800,600)
ventana.show()
sys.exit(app.exec_())