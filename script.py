#importar librerías
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
if  path.exists('fvecto'):
    fvecto = open('fvecto', 'rb') #abre el archivo en modo lectura
    vectorizer = load(fvecto) #carga el archivo en la variable 
else:
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)


def mostrar_menu_principal():
    #os.system('cls') #limpiar la pantalla del terminal
    print("\t**********************************************")
    print("\t***  Análisis de Sentimientos en Twitter   ***")
    print("\t**********************************************")
    print("\t [1] Analizar un bloque de tweets")
    print("\t [2] Analizar un tweet individual")
    print("\t [3] Entrenar algoritmos")
    print("\t [q] Salir")
    return input("Selecciona la opción que quieras: ")

'''
Función para cargar datos de twitter directamente, lo almacena en una base de datos
y lo devuelve en un dataframe. Se usa sólo para ver los resultados. No para entrenar.
Estaría bien poder pasarle a la función el número de tweets que queremos buscar 
por cada iteración, el número de iteraciones y la búsqueda que queremos hacer.
'''
def cargar_datos_de_twitter(consulta, iteraciones, tweet_por_iteracion, vectorizer):
    #Claves y tokens de la cuenta de twitter
    consumer_key='ynSB0dFvqPl3xRU7AmYk39rGT'
    consumer_secret='6alIXTKSxf0RE57QK3fDQ8dxdvlsVr1IRsHDZmoSlMx96YKBFD'
    access_token='966591013182722049-BVXW14Hf5s6O2oIwS3vtJ3S3dOsKLbY'
    access_token_secret='829DTKPjmwsSytmp1ky9fMCJkjV0LZ04TbL9oqHGV6cDm'
    #parámetros de la consulta
    q = 'premier league -filter:retweets AND -filter:replies'
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
    number_of_pages = 1
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
    df['tweet_source'] = df['source'].apply(lambda x: BeautifulSoup(x).get_text())
    devices = list(set(df[df['tweet_source'].str.startswith('Twitter')]['tweet_source']))
    #devices.remove('Twitter Ads')
    df = df[df['tweet_source'].isin(devices)]
    ##################################################################
    #DEL DATAFRAME A UN CSV A UN DATAFRAME
    df2 = pd.DataFrame(data=df['text'])
    df2.to_csv('export.csv')
    dataset = pd.read_csv('export.csv')
    df3 =pd.DataFrame(data=dataset['text'])
    dfNER = pd.DataFrame(data=dataset['text'])
    tweetys = df3['text'] #para NER
    ##################################################################
    #TOKENIZATION inicial para NER
    dfNER['tokens'] = dfNER['text'].apply(TweetTokenizer().tokenize)
    #STOPWORDS
    stopwords_vocabulary = stopwords.words('english') #estará en español?
    dfNER['stopwords'] = dfNER['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
    #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
    punctuations = list(string.punctuation)
    dfNER['punctuation'] = dfNER['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
    dfNER['digits'] = dfNER['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
    dfNER['final'] = dfNER['digits'].apply(lambda x: [i for i in x if len(i) > 1])
    
    #test_data = dfNER['final'][-100:] #saca sólo los últimos 100
    anNER = dfNER['final'][-5:] #saca sólo los últimos 100
    resultadoNER = usar_NER(anNER,3)
    for ent in resultadoNER:
        for it in ent:
            print(it)
            df3 = df3[df3['text'].str.contains(it[0])]
            #TOKENIZATION
            df3['tokens'] = df3['text'].apply(TweetTokenizer().tokenize)
            #STOPWORDS
            stopwords_vocabulary = stopwords.words('english') #estará en español?
            df3['stopwords'] = df3['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
            #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
            punctuations = list(string.punctuation)
            df3['punctuation'] = df3['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
            df3['digits'] = df3['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
            df3['final'] = df3['digits'].apply(lambda x: [i for i in x if len(i) > 1])
            
            test_data = df3['final'][-100:] #saca sólo los últimos 100
            test_data = list(test_data.apply(' '.join))

            test_vectors = vectorizer.transform(test_data)
            mostrar_graph(predecir_Naive_Bayes(test_vectors, it),predecir_SVC(test_vectors, it), predecir_KNN(test_vectors, it), predecir_MLP(test_vectors, it))

            df3 =pd.DataFrame(data=dataset['text'])#vuelve a recargar el df3


    
    
'''
Función que recibe un dataframe con tweets de twitter y los deja preparados para
ser analizados.
'''
def limpieza_de_datos_de_twitter(df):
    df['tweet_source'] = df['source'].apply(lambda x: BeautifulSoup(x).get_text())
    devices = list(set(df[df['tweet_source'].str.startswith('Twitter')]['tweet_source']))
    #devices.remove('Twitter Ads')
    df = df[df['tweet_source'].isin(devices)]
    #TOKENIZATION
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
'''
Función que entrena todos los algoritmos utilizando datos de ficheros de entrenamiento
y de test. En la misma función se limpian los datos tokenizados.
'''
def entrenar_algoritmos(vectorizer):
    filename2 = input("\tEscribe el nombre del fichero de entrenamiento: ") or 'prueba.csv'
    filename = input("\tEscribe el nombre del fichero de pruebas (test): ") or 'testdata.manual.2009.06.14.csv'
    dataset = pd.read_csv(filename)
    tweetys = dataset['final']
    prueba = pd.read_csv(filename) #para la última parte del NER combinado
    dataset2 = pd.read_csv(filename2)
    print("1. Carga de archivos realizada")
    #LIMPIEZA DE DATOS DE DATASET
    #TOKENIZATION
    dataset['tokens'] = dataset['final'].apply(TweetTokenizer().tokenize)
    #STOPWORDS
    stopwords_vocabulary = stopwords.words('english') #estará en español?
    dataset['stopwords'] = dataset['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
    #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
    punctuations = list(string.punctuation)
    dataset['punctuation'] = dataset['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
    dataset['digits'] = dataset['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
    dataset['final'] = dataset['digits'].apply(lambda x: [i for i in x if len(i) > 1])
    print("2. Limpieza del dataset realizada")
    #LIMPIEZA DE DATOS DE DATASET2
    #TOKENIZATION
    dataset2['tokens'] = dataset2['final'].apply(TweetTokenizer().tokenize)
    #STOPWORDS
    stopwords_vocabulary = stopwords.words('english') #estará en español?
    dataset2['stopwords'] = dataset2['tokens'].apply(lambda x: [i for i in x if i.lower() not in stopwords_vocabulary])
    #SPECIAL CHARACTERS AND STOPWORDS REMOVAL
    punctuations = list(string.punctuation)
    dataset2['punctuation'] = dataset2['stopwords'].apply(lambda x: [i for i in x if i not in punctuations])
    dataset2['digits'] = dataset2['punctuation'].apply(lambda x: [i for i in x if i[0] not in list(string.digits)])
    dataset2['final'] = dataset2['digits'].apply(lambda x: [i for i in x if len(i) > 1])
    print("3. Limpieza del dataset2 realizada")
    #Aquí es el lugar donde defino el número de tweets que usaré en los modelos siempre con el porcentaje 80:20
    train_data = dataset2['final'][0:300]
    train_labels = dataset2['label'][0:300]

    test_data = dataset['final'][0:75]
    test_labels = dataset['label'][0:75]

    train_data = list(train_data.apply(' '.join))
    test_data = list(test_data.apply(' '.join))
    #Preparar datos para los modelos
    
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    fvecto = open('fvecto', 'wb')
    dump(vectorizer, fvecto)

    modelos = ['NaiveBayes','Svc','Knn','Tree','Mlp']
    #Vectores para el análisis:

    #ENTRENAMIENTO DE ALGORITMOS
    for alg in modelos:
        if alg == 'NaiveBayes':
            mod = MultinomialNB()
        elif alg == 'Svc':
            mod = SVC()
        elif alg == 'Knn':
            mod = KNeighborsClassifier()
        elif alg == 'Tree':
            mod = DecisionTreeClassifier(random_state=0)
        elif alg == 'Mlp':
            mod = MLPClassifier()
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
'''
Función que analiza un bloque de Tweets. Te da a elegir entre tweets sacados directamente
desde twitter.com o tweets de un archivo.
'''
def analizar_bloque_de_tweets(vectorizer):
    print("\t [1] Sacar tweets de Twitter.com")
    print("\t [2] Sacar tweets de un fichero")
    elect = input("Selecciona la opción que quieras: ")
    if elect == '1':
        cargar_datos_de_twitter('premier league -filter:retweets AND -filter:replies',1,100,vectorizer)
    elif elect == '2':
        filename = input("\tEscribe el nombre del fichero donde se encuentra el bloque de tweets: ") or 'bloque.csv'
        dataset = pd.read_csv(filename)
        tweetys = dataset['final']
        prueba = pd.read_csv(filename) #para la última parte del NER combinado
        #Análisis inicial de NER ################################################
        st = StanfordNERTagger(r'C:\Users\Servicio Técnico\Documents\stanford-ner-2018-02-27\classifiers\english.all.3class.distsim.crf.ser.gz')
        #acuérdate de que cambia para el mac que es donde vas a realizar la presentación
        entities = []
        for r in tweetys:
            #print("está analizando(r): ", r)
            lst_tags = st.tag(r) #no tengo que hacer el split porque ya está hecho?
            for tup in lst_tags:
                #print("está analizando(tup): ", tup)
                if(tup[1] != 'O'):
                    #print("mete(tup) ", tup, "en las entidades")
                    entities.append(tup)
        df_entities = pd.DataFrame(entities)
        df_entities.columns = ["word","ner"]
        #Organizaciones
        organizations =df_entities[df_entities['ner'].str.contains("ORGANIZATION")]
        cnt = Counter(organizations['word'])
        cnt.most_common(3)
        #Personas
        person =df_entities[df_entities['ner'].str.contains("PERSON")]
        cnt_person = Counter(person['word'])
        cnt_person.most_common(3)
        #Localizaciones
        locations =df_entities[df_entities['ner'].str.contains("LOCATION")]
        cnt_location = Counter(locations['word'])
        cnt_location.most_common(3)

def predecir_Naive_Bayes(test_vectors, it):
    mod = MultinomialNB()
    file = open('NaiveBayes', 'rb')
    mod = load(file)
    result = mod.predict(test_vectors)
    pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
    neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
    neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
    y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
    return (it[0],y)

def predecir_SVC(test_vectors, it):
    mod = SVC()
    file = open('Svc', 'rb')
    mod = load(file)
    result = mod.predict(test_vectors)
    pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
    neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
    neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
    y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
    return (it[0],y)

def predecir_KNN(test_vectors, it):
    mod = KNeighborsClassifier()
    file = open('Knn', 'rb')
    mod = load(file)
    result = mod.predict(test_vectors)
    pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
    neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
    neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
    y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
    return (it[0],y)

def predecir_MLP(test_vectors, it):
    mod = MLPClassifier()
    file = open('Mlp', 'rb')
    mod = load(file)
    result = mod.predict(test_vectors)
    pos = len(result[result == 4]) #guardamos la cantidad de tweets positivos
    neg = len(result[result == 0]) #guardamos la cantidad de tweets negativos
    neu = len(result[result == 2]) #guardamos la cantidad de tweets neutros
    y = [pos, neu, neg] # vector de la cantidad de tweets positivos, negativos y neutros
    return (it[0],y)

def mostrar_graph(NB,SVC, KNN, MLP):
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

def usar_NER(tweetys, n):
    st = StanfordNERTagger(r'C:\Users\Servicio Técnico\Documents\stanford-ner-2018-02-27\classifiers\english.all.3class.distsim.crf.ser.gz')
    #acuérdate de que cambia para el mac que es donde vas a realizar la presentación
    entities = []

    for r in tweetys:
        print("está analizando(r): ", r)
        lst_tags = st.tag(r) #no tengo que hacer el split porque ya está hecho?
        for tup in lst_tags:
            print("está analizando(tup): ", tup)
            if(tup[1] != 'O'):
                print("mete(tup) ", tup, "en las entidades")
                entities.append(tup)
    df_entities = pd.DataFrame(entities)
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

###################################################################################
###################################################################################
#Programa principal
choice = mostrar_menu_principal()
if choice == '1':
    print("\nAnalizar un bloque de tweets\n")
    analizar_bloque_de_tweets(vectorizer)
elif choice == '2':
    print("\nAnalizar un tweet individual\n")
    data = {'col_1': ['cabeza de toro', 'rabo de toro', 'caca de vaca', 'toro asco'],'col_2': ['misión cristiana', 'pata de toro', 'cenicienta', 'dime que si']}
    comp = 'toro'
    ejemplo = pd.DataFrame(data)
    ejemplo = ejemplo[ejemplo['col_1'].str.contains(comp)]
    print(ejemplo)
elif choice == '3':
    os.system('cls') #limpiar la pantalla del terminal
    print("\nEntrenar algoritmos\n")
    entrenar_algoritmos(vectorizer)
elif choice == 'q':
    print("\nAdios!")
else:
    print("\nSelecciona una opción correcta\n")