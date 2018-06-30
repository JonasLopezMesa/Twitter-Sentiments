#!/usr/bin/env python
#importar librerías
#LIBRERÍAS PARA QT
import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import QApplication, QLineEdit, QInputDialog, QFileDialog, QLabel, QProgressBar, QVBoxLayout, QTableWidgetItem, QHBoxLayout, QBoxLayout, QPushButton, QTableWidget
from PySide2.QtCore import Qt
from PySide2.QtGui import QTextTable
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
import nltk # [JMTORRES] no necesitas esto si vas a importar cada clase individualmente
from nltk.tag import StanfordNERTagger
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
 
class Ventana(QtWidgets.QWidget):

    def __init__(self, vectorizer, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        #Variable para el entrenamiento de algoritmos
        self.vectorizer = vectorizer
        #Botones del menú principal
        self.buttonBloqueFile = QPushButton("&Analizar Bloque desde archivo", self)
        self.buttonBloqueTwitter = QPushButton("&Analizar Bloque desde Twitter.com", self)
        self.buttonUnTweet = QPushButton("&Analizar un Tweet", self)
        self.buttonEntrenar = QPushButton("&Entrenar algoritmos", self)
        #Añadir los botones al layout del menú
        self.layoutMenu =QHBoxLayout()
        self.layoutMenu.addWidget(self.buttonBloqueFile)
        self.layoutMenu.addWidget(self.buttonBloqueTwitter)
        self.layoutMenu.addWidget(self.buttonUnTweet)
        self.layoutMenu.addWidget(self.buttonEntrenar)
        #Layout donde irían los resultados
        self.layoutWidget =QVBoxLayout()
        self.infoLayout = QVBoxLayout() #Layout que muestra todos los widgets de mostrar un sólo tweet

        #Variables para mostrar un sólo tweet
        self.tituloLabel = QLabel(self) #Etiqueta que muestra el título
        self.infoLayout.addWidget(self.tituloLabel)
        self.tweetLabel = QLabel(self) #Etiqueta que muestra el texto del tweet
        self.infoLayout.addWidget(self.tweetLabel)
        self.nerLabel = QLabel(self) #Etiqueta que muestra el título de NER
        self.infoLayout.addWidget(self.nerLabel)
        self.nonerLabel = QLabel(self) #Etiqueta que muestra el aviso de que no hay NER
        self.infoLayout.addWidget(self.nonerLabel)
        self.tabla = QTableWidget() #Tabla que muestra los resultados del NER
        self.infoLayout.addWidget(self.tabla)
        self.claLabel = QLabel(self) #Etiqueta que muestra el título que precede a la tabla de sentimientos
        self.infoLayout.addWidget(self.claLabel)
        self.tablaSent = QTableWidget() #Tabla que muestra los sentimientos de cada algoritmo
        self.infoLayout.addWidget(self.tablaSent) 
        #Variables para mostrar en entrenamiento
        self.entrenamientoLabel = QLabel(self)
        #Variables para la barra de progreso
        self.progressBarUnTweet = QProgressBar(self)
        self.progressLayout = QVBoxLayout()
        self.progresLabel = QLabel(self)
        self.progressLayout.addWidget(self.progresLabel)
        self.progressLayout.addWidget(self.progressBarUnTweet)
        #Variables para la selección en cargar datos de twitter
        self.consulta = QLineEdit()
        self.buttonConsulta = QPushButton()
        self.consultaLayout = QHBoxLayout()
        self.consultaLayout.addWidget(self.consulta)
        self.consultaLayout.addWidget(self.buttonConsulta)
        self.consultaText = ""
        self.consultaTweet = 0
        self.nerCantidadValor = 0
        # Conectar botón de consulta
        self.connect(self.buttonConsulta, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("guardarTexto()"))

        #diálogo de archivo
        self.dialogo1 = QFileDialog(self)

        #Creación del layout principal que anida el resto de layouts
        self.layoutPrincipal = QVBoxLayout()
        self.layoutPrincipal.addLayout(self.layoutMenu)
        self.layoutPrincipal.addLayout(self.progressLayout)
        self.layoutPrincipal.addStretch()
        self.layoutPrincipal.addLayout(self.layoutWidget)

        self.setLayout(self.layoutPrincipal)


        #Diálogo para configurar parámetros de bloque de Twitter
        self.dialogConsulta = QInputDialog(self)
        self.dialogTweets = QInputDialog(self)
        self.nerCantidad = QInputDialog(self)
        

        # Conectar a analizarUnTweet:
        self.buttonUnTweet.clicked.connect(self.analizarUnTweet)
        # Conectar a entrenar_algoritmos
        self.buttonEntrenar.clicked.connect(self.entrenar_algoritmos)
        # Conectar a cargar datos de Twitter
        self.buttonBloqueTwitter.clicked.connect(self.cuadroDialogo)
        # Conectar a cargar datos de archivo
        self.buttonBloqueFile.clicked.connect(self.analizarDeArchivo)

    def limpiarLayout(self,layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().hide()
    def mostrarLayout(self,layout):
        for i in range(layout.count()): 
            layout.itemAt(i).widget().show()

    def cuadroDialogo(self):
        self.consultaText = self.dialogConsulta.getText(self,"Consulta de Twitter", "¿Sobre qué quieres buscar?")
        print(self.consultaText[0])
        self.consultaTweets = self.dialogTweets.getInt(self,"Cuántos twits quieres usar","La cantidad se multiplica por 100")
        print(self.consultaTweets)
        self.nerCantidadValor = self.nerCantidad.getInt(self,"Cuántos twits quieres usar en NER","Tweets que se usarán para NER")
        print(self.nerCantidadValor)
        self.cargar_datos_de_twitter()
    '''Función para cargar datos de twitter directamente, lo almacena en una base de datos
    y lo devuelve en un dataframe. Se usa sólo para ver los resultados. No para entrenar.
    Estaría bien poder pasarle a la función el número de tweets que queremos buscar 
    por cada iteración, el número de iteraciones y la búsqueda que queremos hacer.'''
    def cargar_datos_de_twitter(self):
        self.limpiarLayout(self.infoLayout)
        consumer_key='ynSB0dFvqPl3xRU7AmYk39rGT'
        consumer_secret='6alIXTKSxf0RE57QK3fDQ8dxdvlsVr1IRsHDZmoSlMx96YKBFD'
        access_token='966591013182722049-BVXW14Hf5s6O2oIwS3vtJ3S3dOsKLbY'
        access_token_secret='829DTKPjmwsSytmp1ky9fMCJkjV0LZ04TbL9oqHGV6cDm'
        #parámetros de la consulta
        q = self.consultaText[0] + ' -filter:retweets AND -filter:replies'
        #q = 'premier league -filter:retweets AND -filter:replies'
        print(q)
        url = 'https://api.Twitter.com/1.1/search/tweets.json'
        pms = {'q' : q, 'count' : 100, 'lang' : 'en', 'result_type': 'recent'} 
        auth = OAuth1(consumer_key, consumer_secret, access_token,access_token_secret)
        #inicialización de la base de datos para cargar los datos
        database_name = "baseDeDatos"
        collection_name = "coleccion"
        client = MongoClient('mongodb://localhost:27017/')
        db = client[database_name]
        collection = db[collection_name]
        #Paginación (carga de 100 en 100 datos)
        pages_counter = 0
        number_of_pages = self.consultaTweets[0]
        while pages_counter < number_of_pages:
            pages_counter += 1
            res = requests.get(url, params = pms, auth=auth)
            print("Connection status: %s" % res.reason)
            tweets = res.json()
            ids = [i['id'] for i in tweets['statuses']]
            pms['max_id'] = min(ids) - 1
            collection.insert_many(tweets['statuses'])
        #Pasar de la base de datos a un dataframe
        ##################################################################
        documents = []
        for doc in collection.find():
            documents.append(doc)
        df = pd.DataFrame(documents)
        #Limpieza de datos
        df = self.limpieza_de_datos_de_twitter(df)
        df2 = pd.DataFrame(data=df['text'][-100:])
        dfNER = pd.DataFrame(data=df['text'])
        dfNER = self.tokenizar(dfNER)
        anNER = dfNER['final'][-self.nerCantidadValor[0]:] #saca sólo los últimos 5
        resultadoNER = self.usar_NER(anNER,3)
        for ent in resultadoNER:
            for it in ent:
                print(it)
                df2 = df2[df2['text'].str.contains(it[0])]
                df2 = self.tokenizar(df2)
                test_data = df2['final'][-100:] #saca sólo los últimos 100
                test_data = list(test_data.apply(' '.join))
                test_vectors = self.vectorizer.transform(test_data)
                self.mostrar_graph(self.predecir_Naive_Bayes(test_vectors, it),self.predecir_SVC(test_vectors, it), self.predecir_KNN(test_vectors, it), self.predecir_MLP(test_vectors, it))
                df2 = pd.DataFrame(data=df['text'])#vuelve a recargar el df3
        self.progresLabel.setText("FINALIZADO")
    '''Función que analiza twits sacados de un archivo a elegir por el usuario'''
    def analizarDeArchivo(self):
        filename = self.dialogo1.getOpenFileName(self, "Selecciona el fichero a analizar","/")
        filename = filename[0].split("/")
        filename = filename[-1]
        self.nerCantidadValor = self.nerCantidad.getInt(self,"Cuántos twits quieres usar en NER","Tweets que se usarán para NER")
        self.progressBarUnTweet.reset()
        self.progressBarUnTweet.setMaximum(100)
        self.progressBarUnTweet.setMinimum(0)
        #filename = input("\tEscribe el nombre del fichero donde se encuentra el bloque de tweets: ") or 'bloque.csv'
        dataset = pd.read_csv(filename)
        df3 =pd.DataFrame(data=dataset['text'])
        dfNER = pd.DataFrame(data=dataset['text'])
        self.progressBarUnTweet.setValue(20)
        self.progresLabel.setText("Tokenizando")
        dfNER = self.tokenizar(dfNER)
        anNER = dfNER['final'][-self.nerCantidadValor[0]:]
        self.progressBarUnTweet.setValue(30)
        resultadoNER = self.usar_NER(anNER,3)
        self.progressBarUnTweet.reset()
        self.progressBarUnTweet.setMaximum(100)
        self.progressBarUnTweet.setMinimum(0)
        self.progressBarUnTweet.setValue(90)
        self.progresLabel.setText("Dibujando gráficos")
        for ent in resultadoNER:
            for it in ent:
                print(it)
                df3 = df3[df3['text'].str.contains(it[0])]
                df3 = self.tokenizar(df3)
                test_data = df3['final'][-100:] #saca sólo los últimos 100
                test_data = list(test_data.apply(' '.join))
                test_vectors = self.vectorizer.transform(test_data)
                self.mostrar_graph(self.predecir_Naive_Bayes(test_vectors, it),self.predecir_SVC(test_vectors, it), self.predecir_KNN(test_vectors, it), self.predecir_MLP(test_vectors, it))
                df3 =pd.DataFrame(data=dataset['text'])#vuelve a recargar el df3
        self.progressBarUnTweet.setValue(100)
        self.progresLabel.setText("FINALIZADO")
    '''Función que analiza un tweet individual sacado de twitter.com'''
    def analizarUnTweet(self):
        #self.layoutWidget.insertLayout(2,self.infoLayout)
        #Barra de progreso
        #self.infoLayout.show()
        self.consultaText = self.dialogConsulta.getText(self,"Consulta de Twitter", "¿Sobre qué quieres buscar?")
        self.progressBarUnTweet.reset()
        self.progressBarUnTweet.setMaximum(10)
        self.progressBarUnTweet.setMinimum(0)
        #self.layoutProgressBar.addWidget(self.progressBarUnTweet)
        #Claves y tokens de la cuenta de twitter
        consumer_key='ynSB0dFvqPl3xRU7AmYk39rGT'
        consumer_secret='6alIXTKSxf0RE57QK3fDQ8dxdvlsVr1IRsHDZmoSlMx96YKBFD'
        access_token='966591013182722049-BVXW14Hf5s6O2oIwS3vtJ3S3dOsKLbY'
        access_token_secret='829DTKPjmwsSytmp1ky9fMCJkjV0LZ04TbL9oqHGV6cDm'
        self.progressBarUnTweet.setValue(1)
        self.progresLabel.setText("Estableciendo claves de acceso")
        #parámetros de la consulta
        q = self.consultaText[0] + ' -filter:retweets AND -filter:replies'
        url = 'https://api.Twitter.com/1.1/search/tweets.json'
        pms = {'q' : q, 'count' : 10, 'lang' : 'en', 'result_type': 'recent'} 
        auth = OAuth1(consumer_key, consumer_secret, access_token,access_token_secret)
        #inicialización de la base de datos para cargar los datos
        database_name = "baseDeDatos"
        collection_name = "coleccion"
        client = MongoClient('mongodb://localhost:27017/')
        db = client[database_name]
        collection = db[collection_name]
        self.progressBarUnTweet.setValue(2)
        self.progresLabel.setText("Iniciando base de datos")
        #Paginación (carga de 100 en 100 datos)
        pages_counter = 0
        number_of_pages = 1 #Número de veces que cuenta
        while pages_counter < number_of_pages:
            pages_counter += 1
            res = requests.get(url, params = pms, auth=auth)
            print("Connection status: %s" % res.reason)
            tweets = res.json()
            ids = [i['id'] for i in tweets['statuses']]
            pms['max_id'] = min(ids) - 1
            collection.insert_many(tweets['statuses'])
        self.progressBarUnTweet.setValue(3)
        self.progresLabel.setText("Guardando tweets en base de datos")
        #Pasar de la base de datos a un dataframe
        ##################################################################
        documents = []
        for doc in collection.find():
            documents.append(doc)
        df = pd.DataFrame(documents)
        mostrar = pd.DataFrame(documents)
        self.progressBarUnTweet.setValue(4)
        #Limpieza de datos
        df = self.limpieza_de_datos_de_twitter(df)
        df2 = pd.DataFrame(data=df['text'][-1:])
        dfNER = pd.DataFrame(data=df['text'][-1:])
        tweet = mostrar['text'][-1:]
        dfNER = self.tokenizar(dfNER)
        anNER = dfNER['final'][-1:]
        print(anNER)
        resultadoNER = self.usar_NER(anNER,10)
        self.progressBarUnTweet.setValue(5)
        self.progresLabel.setText("Limpiando datos")

        df2 = self.tokenizar(df2)
        test_data = df2['final']
        test_data = list(test_data.apply(' '.join))
        test_vectors = self.vectorizer.transform(test_data)
        self.progressBarUnTweet.setValue(6)
        self.progresLabel.setText("Transformando datos datos")

        nb = self.predecir_Naive_Bayes(test_vectors, 'nada')
        self.progressBarUnTweet.setValue(7)
        self.progresLabel.setText("Naive Bayes")
        svc = self.predecir_SVC(test_vectors, 'nada')
        self.progressBarUnTweet.setValue(8)
        self.progresLabel.setText("SVC")
        knn = self.predecir_KNN(test_vectors, 'nada')
        self.progressBarUnTweet.setValue(9)
        self.progresLabel.setText("KNN")
        mlp = self.predecir_MLP(test_vectors, 'nada')
        self.progressBarUnTweet.setValue(10)
        self.progresLabel.setText("FINALIZADO")

        #Mostrar la información###########################################################
        #For para limpiar el layout
        #for i in reversed(range(self.layoutWidget.count())): 
            #self.removeItem(self.layoutWidget.itemAt(i))

        #self.tituloLabel = QLabel(self)
        self.tituloLabel.setText("<h1>ANÁLISIS DE SENTIMIENTOS EN UN TWEET INDIVIDUAL</h1>")
        
        #self.layoutWidget.insertStretch(1)
        
        
        #tweetLabel = QLabel(self)
        self.tweetLabel.setText("<b>TWEET:</b> " + tweet.to_string())
        
        #nerLabel = QLabel(self)
        self.nerLabel.setText("<h2>Entidades</h2>")
        
        if resultadoNER == 0:
            #nonerLabel = QLabel(self)
            self.nonerLabel.setText("<b><font size=" + "3" + " color=" + "red" + ">NO SE RECONOCIERON LAS ENTIDADES</font></b>")
            self.tabla.clear()
            
        else:
            #tabla = QTableWidget()
            self.nonerLabel.setText(" ")
            self.tabla.setVerticalHeaderLabels(["prueba","prueba2"])
            self.tabla.horizontalHeader().hide()
            self.tabla.verticalHeader().hide()
            self.tabla.setColumnCount(1)
            fila = 0
            filasTotales = 0
            for i in resultadoNER:
                filasTotales = len(i)+filasTotales
                self.tabla.setRowCount(filasTotales)
                for j in i:
                    print("\t - ", j[1])
                    columna1 = QTableWidgetItem(j[0])
                    self.tabla.setItem(fila,0,columna1)
                    fila = fila + 1
            
        #claLabel = QLabel(self)
        self.claLabel.setText("<h2>Clasificación de sentimientos</h2>")
        
        #self.tablaSent = QTableWidget()
        self.tablaSent.setColumnCount(2)
        self.tablaSent.setRowCount(4)
        self.tablaSent.horizontalHeader().hide()
        self.tablaSent.verticalHeader().hide()
        self.tablaSent.setVerticalHeaderLabels(['Clasificador','Resultado'])
        self.tablaSent.setItem(0,0,QTableWidgetItem("Naive Bayes"))
        print("Sentimiento con Naive Bayes")
        if nb[1][0] == 1:
            self.tablaSent.setItem(0,1,QTableWidgetItem("Positivo"))
        elif nb[1][1] == 1:
            self.tablaSent.setItem(0,1,QTableWidgetItem("Neutro"))
        elif nb[1][2] == 1:
            self.tablaSent.setItem(0,1,QTableWidgetItem("Negativo"))
        self.tablaSent.setItem(1,0,QTableWidgetItem("Clasificador SVC"))
        print("\tSentimiento con Clasificador SVC")
        if svc[1][0] == 1:
            self.tablaSent.setItem(1,1,QTableWidgetItem("Positivo"))
        elif svc[1][1] == 1:
            self.tablaSent.setItem(1,1,QTableWidgetItem("Neutro"))
        elif svc[1][2] == 1:
            self.tablaSent.setItem(1,1,QTableWidgetItem("Negativo"))
        self.tablaSent.setItem(2,0,QTableWidgetItem("Clasificador K-Neighbors"))
        print("\tSentimiento con Clasificador K-Neighbors")
        if knn[1][0] == 1:
            self.tablaSent.setItem(2,1,QTableWidgetItem("Positivo"))
        elif knn[1][1] == 1:
            self.tablaSent.setItem(2,1,QTableWidgetItem("Neutro"))
        elif knn[1][2] == 1:
            self.tablaSent.setItem(2,1,QTableWidgetItem("Negativo"))
        self.tablaSent.setItem(3,0,QTableWidgetItem("Clasificador MLP"))
        print("\tSentimiento con Clasificador MLP")
        if mlp[1][0] == 1:
            self.tablaSent.setItem(3,1,QTableWidgetItem("Positivo"))
        elif mlp[1][1] == 1:
            self.tablaSent.setItem(3,1,QTableWidgetItem("Neutro"))
        elif mlp[1][2] == 1:
            self.tablaSent.setItem(3,1,QTableWidgetItem("Negativo"))
        
        # [JMTORRES] ojo con lo que haces aquí. Si consume mucho tiempo de CPU la aplicación
        # parecerá bloqueada. Eso habría que tenerlo en cuenta si pasa.
        self.layoutWidget.addLayout(self.infoLayout)
        self.mostrarLayout(self.infoLayout)
    '''Función que tokeniza los datos de un tweet, eliminando las stopwords y los carácteres
    especiales'''
    def tokenizar(self, df):
        #TOKENIZATION inicial para NER
        df['tokens'] = df['text'].apply(TweetTokenizer().tokenize)
        #STOPWORDS
        stopwords_vocabulary = stopwords.words('english') #estará en español?
        df['stopwords'] = df['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
        #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
        punctuations = list(string.punctuation)
        df['punctuation'] = df['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
        df['digits'] = df['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
        df['final'] = df['digits'].apply(lambda x: [i for i in x if len(i) > 1])
        return df
    '''Función que recibe un dataframe con tweets de twitter y los deja preparados para
    ser tokenizados'''
    def limpieza_de_datos_de_twitter(self, df):
        df['tweet_source'] = df['source'].apply(lambda x: BeautifulSoup(x).get_text())
        devices = list(set(df[df['tweet_source'].str.startswith('Twitter')]['tweet_source']))
        #devices.remove('Twitter Ads')
        df = df[df['tweet_source'].isin(devices)]
        return df
    '''Funciones de predicción de los diferentes algoritmos para los diferentes modelos'''
    def predecir_Naive_Bayes(self, test_vectors, it):
        mod = MultinomialNB()
        file = open('NaiveBayes', 'rb')
        mod = load(file)
        result = mod.predict(test_vectors)
        pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
        neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
        neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
        y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
        return (it[0],y)
    def predecir_SVC(self, test_vectors, it):
        mod = SVC()
        file = open('Svc', 'rb')
        mod = load(file)
        result = mod.predict(test_vectors)
        pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
        neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
        neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
        y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
        return (it[0],y)
    def predecir_KNN(self, test_vectors, it):
        mod = KNeighborsClassifier()
        file = open('Knn', 'rb')
        mod = load(file)
        result = mod.predict(test_vectors)
        pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
        neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
        neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
        y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
        return (it[0],y)
    def predecir_MLP(self, test_vectors, it):
        mod = MLPClassifier()
        file = open('Mlp', 'rb')
        mod = load(file)
        result = mod.predict(test_vectors)
        pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
        neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
        neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
        y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
        return (it[0],y)
    '''Función que muestra los gráficos utilizando los plots de python'''
    def mostrar_graph(self, NB,SVC, KNN, MLP):
        #Naive Bayes
        plt.subplot(221)
        plt.title("NB para la entidad " + NB[0])
        plt.ylabel('tweets')
        plt.xticks(range(len(NB[1])), ['positive', 'neutral', 'negative'])
        plt.bar(range(len(NB[1])), height=NB[1], width = 0.75, align = 'center', alpha = 0.8)
        #SVC
        plt.subplot(222)
        plt.title("SVC para la entidad " + SVC[0])
        plt.ylabel('tweets')
        plt.xticks(range(len(SVC[1])), ['positive', 'neutral', 'negative'])
        plt.bar(range(len(SVC[1])), height=SVC[1], width = 0.75, align = 'center', alpha = 0.8)
        #KNN
        plt.subplot(223)
        plt.title("KNN para la entidad " + KNN[0])
        plt.ylabel('tweets')
        plt.xticks(range(len(KNN[1])), ['positive', 'neutral', 'negative'])
        plt.bar(range(len(KNN[1])), height=KNN[1], width = 0.75, align = 'center', alpha = 0.8)
        #MLP
        plt.subplot(224)
        plt.title("MLP para la entidad " + MLP[0])
        plt.ylabel('tweets')
        plt.xticks(range(len(MLP[1])), ['positive', 'neutral', 'negative'])
        plt.bar(range(len(MLP[1])), height=MLP[1], width = 0.75, align = 'center', alpha = 0.8)
        plt.show()
    '''Función que utiliza NER para detectar entidades.'''
    def usar_NER(self, tweetys, n):
        self.progressBarUnTweet.reset()
        self.progressBarUnTweet.setMaximum(len(tweetys)*10)
        self.progressBarUnTweet.setMinimum(0)
        self.progresLabel.setText("ANALIZANDO ENTIDADES")
        #st = StanfordNERTagger(r'C:\Users\Servicio Técnico\Documents\stanford-ner-2018-02-27\classifiers\english.all.3class.distsim.crf.ser.gz')
        st = StanfordNERTagger('/Users/jonas/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz')
        #acuérdate de que cambia para el mac que es donde vas a realizar la presentación
        entities = []
        tindice = 0
        for r in tweetys:
            print("está analizando(r): ", r)
            lst_tags = st.tag(r) #no tengo que hacer el split porque ya está hecho?
            for tup in lst_tags:
                print("está analizando(tup): ", tup)
                self.progressBarUnTweet.setValue(tindice)
                tindice = tindice + 1
                if(tup[1] != 'O'):
                    print("mete(tup) ", tup, "en las entidades")
                    entities.append(tup)
        df_entities = pd.DataFrame(entities)
        self.progressBarUnTweet.setValue(len(tweetys)*10)
        if df_entities.size >0:
            df_entities.columns = ["word","ner"]
            #Organizaciones
            organizations =df_entities[df_entities['ner'].str.contains("ORGANIZATION")]
            cnt = Counter(organizations['word'])
            organizaciones = cnt.most_common(n)
            #Personas
            person =df_entities[df_entities['ner'].str.contains("PERSON")]
            cnt_person = Counter(person['word'])
            personas = cnt_person.most_common(n)
            #Localizaciones
            locations =df_entities[df_entities['ner'].str.contains("LOCATION")]
            cnt_location = Counter(locations['word'])
            lugares = cnt_location.most_common(n)
            return (organizaciones, personas, lugares)
        else:
            return 0
    '''Función que entrena todos los algoritmos utilizando datos de ficheros de entrenamiento
    y de test. En la misma función se limpian los datos tokenizados. Al final detecta cuál
    es la mejor configuración para el algoritmo y los entrena con dicha configuración'''
    def entrenar_algoritmos(self):
        #self.infoLayout.hide()
        self.progressBarUnTweet.reset()
        self.progressBarUnTweet.setMaximum(438)
        self.progressBarUnTweet.setMinimum(0)
        filename2 = self.dialogo1.getOpenFileName(self, "Selecciona el fichero de entrenamiento","/")
        filename2 = filename2[0].split("/")
        filename2 = filename2[-1]
        filename = self.dialogo1.getOpenFileName(self, "Selecciona el fichero de pruebas", "/")
        filename = filename[0].split("/")
        filename = filename[-1]
        dataset = pd.read_csv(filename)
        tweetys = dataset['text']
        prueba = pd.read_csv(filename) #para la última parte del NER combinado
        dataset2 = pd.read_csv(filename2)
        print("1. Carga de archivos realizada")
        #CLEANING DATASET
        #TOKENIZATION
        dataset['tokens'] = dataset['text'].apply(TweetTokenizer().tokenize)
        #STOPWORDS
        stopwords_vocabulary = stopwords.words('english')
        dataset['stopwords'] = dataset['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
        #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
        punctuations = list(string.punctuation)
        dataset['punctuation'] = dataset['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
        dataset['digits'] = dataset['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
        dataset['final'] = dataset['digits'].apply(lambda x: [i for i in x if len(i) > 1])
        print("2. Limpieza del dataset realizada")
        self.progressBarUnTweet.setValue(1)
        self.progresLabel.setText("Limpiando datos fichero entrenamiento")
        #CLEANING DATASET2
        #TOKENIZATION
        dataset2['tokens'] = dataset2['text'].apply(TweetTokenizer().tokenize)
        #STOPWORDS
        stopwords_vocabulary = stopwords.words('english') #estará en español?
        dataset2['stopwords'] = dataset2['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
        #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
        punctuations = list(string.punctuation)
        dataset2['punctuation'] = dataset2['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
        dataset2['digits'] = dataset2['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
        dataset2['final'] = dataset2['digits'].apply(lambda x: [i for i in x if len(i) > 1])
        print("3. Limpieza del dataset2 realizada")
        self.progressBarUnTweet.setValue(2)
        self.progresLabel.setText("Limpiando datos fichero de pruebas")
        #Here is the place where we set the number of tweets that we use to models. Always whit 80:20 percent.
        train_data = dataset2['final'][0:500]
        train_labels = dataset2['label'][0:500]

        test_data = dataset['final'][0:125]
        test_labels = dataset['label'][0:125]

        train_data = list(train_data.apply(' '.join))
        test_data = list(test_data.apply(' '.join))
        self.progressBarUnTweet.setValue(3)
        self.progresLabel.setText("Actualizando datos de entrenamiento")
        #Preparing data for models
        
        train_vectors = self.vectorizer.fit_transform(train_data)
        test_vectors = self.vectorizer.transform(test_data)

        fvecto = open('fvecto', 'wb')
        dump(self.vectorizer, fvecto)

        modelos = ['NaiveBayes','Svc','Knn','Mlp']
        #Analisys vectors:
        puntuaciones = [0,0,0,0]

        params_svc = [['linear','poly','tbf','sigmod','precomputed'],[3,5,10],[0.1,0.5,0.9],[True,False],[True,False]]
        best_svc = []
        params_knn = [[1,5,10],['uniform','distance'],['ball_tree','kd_tree','brute','auto'],[5,30,100],[1,2]]
        best_knn = []
        params_mlp = [[50,100,150],['identity','logistic','tanh','relu'],[0.00005,0.0001,0.001],['constant','invscaling','adaptative']]
        best_mlp = []
        self.progressBarUnTweet.setValue(4)
        self.progresLabel.setText("Preparando parámetros de algoritmos")
        print("LLEGA HASTA AQUÍ")
        #TRAINING ALGORITHMs
        progreso = 5
        for alg in modelos:
            if alg == 'Svc':
                print("Svc", progreso)
                for a in params_svc[0]:
                    for b in params_svc[1]:
                        for c in params_svc[2]:
                            for d in params_svc[3]:
                                for e in params_svc[4]:
                                    mod = SVC(kernel=a,degree=b,coef0=c,probability=d,shrinking=e)
                                    punt = self.entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
                                    self.progressBarUnTweet.setValue(progreso)
                                    progreso = progreso + 1
                                    self.progresLabel.setText("Entrenando SVC con kernel " + a)
                                    if punt > puntuaciones[0]:
                                        puntuaciones[0] = punt
                                        best_svc = [a,b,c,d,e]
            elif alg == 'NaiveBayes':
                mod = MultinomialNB()
                puntuaciones[1] = self.entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
            elif alg == 'Knn':
                print("Knn", progreso)
                for a in params_knn[0]:
                    for b in params_knn[1]:
                        for c in params_knn[2]:
                            for d in params_knn[3]:
                                for e in params_knn[4]:
                                    self.progressBarUnTweet.setValue(progreso)
                                    self.progresLabel.setText("Entrenando KNN con kernel " + b + c)
                                    progreso = progreso + 1
                                    mod = KNeighborsClassifier(n_neighbors=a,weights=b,algorithm=c,leaf_size=d,p=e)
                                    punt = self.entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
                                    if punt > puntuaciones[2]:
                                        puntuaciones[2] = punt
                                        best_knn = [a,b,c,d,e]
            elif alg == 'Mlp':
                print("Mlp", progreso)
                for a in params_mlp[0]:
                    for b in params_mlp[1]:
                        for c in params_mlp[2]:
                            for d in params_mlp[3]:
                                self.progressBarUnTweet.setValue(progreso)
                                self.progresLabel.setText("Entrenando MLP con kernel " + b + d)
                                progreso = progreso + 1
                                mod = MLPClassifier(hidden_layer_sizes=a,activation=b,alpha=c,learning_rate=d)
                                punt = self.entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
                                if punt > puntuaciones[3]:
                                    puntuaciones[3] = punt
                                    best_mlp = [a,b,c,d]
        print("PUNTUACIONES: ",puntuaciones)
        print("BEST SVC", best_svc)
        print("BEST MLP", best_mlp)
        print("BEST KNN", best_knn)
        tmp = 0
        guia = 0
        for h in puntuaciones:
            if h > tmp:
                best_model = guia
                tmp = h
            guia = guia + 1
        self.progressBarUnTweet.setValue(progreso)
        progreso = progreso + 1
        self.entrenar('Svc',train_vectors,train_labels,test_vectors, test_labels, SVC(kernel=best_svc[0],degree=best_svc[1],coef0=best_svc[2],probability=best_svc[3],shrinking=best_svc[4]))
        print("1")
        self.entrenar('NaiveBayes',train_vectors,train_labels,test_vectors, test_labels, mod = MultinomialNB())
        print("2")
        self.entrenar('Knn',train_vectors,train_labels,test_vectors, test_labels, KNeighborsClassifier(n_neighbors=best_knn[0],weights=best_knn[1],algorithm=best_knn[2],leaf_size=best_knn[3],p=best_knn[4]))
        print("3")
        self.entrenar('Mlp',train_vectors,train_labels,test_vectors, test_labels, MLPClassifier(hidden_layer_sizes=best_mlp[0],activation=best_mlp[1],alpha=best_mlp[2],learning_rate=best_mlp[3]))
        print("4")
        self.progressBarUnTweet.setValue(progreso)
        self.progresLabel.setText("FINALIZADO")
        self.entrenamientoLabel.setText("<h1>ENTRENAMIENTO REALIZADO CON ÉXITO</h1>")
        self.layoutWidget.addWidget(self.entrenamientoLabel)
    '''Función auxiliar para usar_NER que guarda los archivos de entrenamiento para que
    se guarden en futuras sesiones'''
    def entrenar(self, alg,train_vectors,train_labels,test_vectors, test_labels, mod):
        nfile = alg
        if path.exists(nfile):
            file = open(nfile, 'rb') #abre el archivo en modo lectura
            mod = load(file) #carga el archivo en la variable 
            mod.fit(train_vectors, train_labels).score(test_vectors, test_labels) #lo entrena
            file.close() #cierra el archivo 
            file = open(nfile, 'wb') #abre el archivo en modo escritura
            dump(mod, file) #actualiza el entrenamiento
        else:
            file = open(nfile, 'wb') #abre el archivo en modo escritura
            mod.fit(train_vectors, train_labels).score(test_vectors, test_labels) #lo entrna
            dump(mod, file) #guarda el entrenamiento
        print("MODELO ", alg, " ENTRENADO Y PROBADO")
        print(classification_report(test_labels, mod.predict(test_vectors)))
        print(confusion_matrix(test_labels, mod.predict(test_vectors)))
        predicted = cross_val_predict(mod, test_vectors, test_labels, cv=10)
        print("Cross validation %s" % accuracy_score(test_labels, predicted))
        return accuracy_score(test_labels,predicted)
      
def main():
    """ Función principal del programa
    """
    # [JMTORRES] Las funciones y clases se documenta con """comentario"""
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

    #Datos para el entrenamiento
    if  path.exists('fvecto'):
        fvecto = open('fvecto', 'rb') #abre el archivo en modo lectura
        vectorizer = load(fvecto) #carga el archivo en la variable 
    else:
        vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)

    best_model = 9
      
    app = QtWidgets.QApplication(sys.argv) #variable para la interfaz
    # [JMTORRES] si ventana necesita los vectorizer lo puedes pasar en el consructor.
    ventana = Ventana(vectorizer)
    ventana.resize(800,600)
    ventana.show()
    sys.exit(app.exec_())       # [JMTORRES] La ejecución no sale de exec() hasta que la ventana principal se cierra.

if __name__ == '__main__':
      main()