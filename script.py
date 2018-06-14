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
#Datos para el entrenamiento
if  path.exists('fvecto'):
    fvecto = open('fvecto', 'rb') #abre el archivo en modo lectura
    vectorizer = load(fvecto) #carga el archivo en la variable 
else:
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
best_model = 9
'''
Función que muestra el menú principal
'''
def mostrar_menu_principal():
    os.system('cls') #limpiar la pantalla del terminal
    print("\t**********************************************")
    print("\t***  Análisis de Sentimientos en Twitter   ***")
    print("\t**********************************************")
    if path.exists('fvecto') and path.exists('Knn'):
        print("\tLOS MODELOS ESTÁN ENTRENADOS")
    else:
        print("\tLOS MODELOS [NO] ESTÁN ENTRENADOS")
    if best_model == 0:
        print("\tEl mejor modelo es SVC")
    elif best_model == 1:
        print("\tEl mejor modelo es NB")
    elif best_model == 2:
        print("\tEl mejor modelo es KNN")
    elif best_model == 3:
        print("\tEl mejor modelo es MLP")
    elif best_model == 9:
        pass
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
    df = limpieza_de_datos_de_twitter(df)
    df2 = pd.DataFrame(data=df['text'][-100:])
    dfNER = pd.DataFrame(data=df['text'])
    dfNER = tokenizar(dfNER)
    anNER = dfNER['final'][-5:] #saca sólo los últimos 5
    resultadoNER = usar_NER(anNER,3)
    for ent in resultadoNER:
        for it in ent:
            print(it)
            df2 = df2[df2['text'].str.contains(it[0])]
            df2 = tokenizar(df2)
            test_data = df2['final'][-100:] #saca sólo los últimos 100
            test_data = list(test_data.apply(' '.join))
            test_vectors = vectorizer.transform(test_data)
            mostrar_graph(predecir_Naive_Bayes(test_vectors, it),predecir_SVC(test_vectors, it), predecir_KNN(test_vectors, it), predecir_MLP(test_vectors, it))
            df2 = pd.DataFrame(data=df['text'])#vuelve a recargar el df3
'''
Función que tokeniza los datos de un tweet, eliminando las stopwords y los carácteres
especiales
'''
def tokenizar(df):
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
'''
Función que recibe un dataframe con tweets de twitter y los deja preparados para
ser analizados.
'''
def limpieza_de_datos_de_twitter(df):
    df['tweet_source'] = df['source'].apply(lambda x: BeautifulSoup(x).get_text())
    devices = list(set(df[df['tweet_source'].str.startswith('Twitter')]['tweet_source']))
    #devices.remove('Twitter Ads')
    df = df[df['tweet_source'].isin(devices)]
    return df
'''
Función que entrena todos los algoritmos utilizando datos de ficheros de entrenamiento
y de test. En la misma función se limpian los datos tokenizados.
'''
def entrenar_algoritmos(vectorizer):
    filename2 = input("\tEscribe el nombre del fichero de entrenamiento: ") or 'prueba.csv'
    filename = input("\tEscribe el nombre del fichero de pruebas (test): ") or 'testdata.manual.2009.06.14.csv'
    dataset = pd.read_csv(filename)
    tweetys = dataset['text']
    prueba = pd.read_csv(filename) #para la última parte del NER combinado
    dataset2 = pd.read_csv(filename2)
    print("1. Carga de archivos realizada")
    #LIMPIEZA DE DATOS DE DATASET
    #TOKENIZATION
    dataset['tokens'] = dataset['text'].apply(TweetTokenizer().tokenize)
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
    #Aquí es el lugar donde defino el número de tweets que usaré en los modelos siempre con el porcentaje 80:20
    train_data = dataset2['final'][0:500]
    train_labels = dataset2['label'][0:500]

    test_data = dataset['final'][0:125]
    test_labels = dataset['label'][0:125]

    train_data = list(train_data.apply(' '.join))
    test_data = list(test_data.apply(' '.join))
    #Preparar datos para los modelos
    
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    fvecto = open('fvecto', 'wb')
    dump(vectorizer, fvecto)

    modelos = ['NaiveBayes','Svc','Knn','Mlp']
    #Vectores para el análisis:
    puntuaciones = [0,0,0,0]

    params_svc = [['linear','poly','tbf','sigmod','precomputed'],[3,5,10],[0.1,0.5,0.9],[True,False],[True,False]]
    best_svc = []
    params_knn = [[1,5,10],['uniform','distance'],['ball_tree','kd_tree','brute','auto'],[5,30,100],[1,2]]
    best_knn = []
    params_mlp = [[50,100,150],['identity','logistic','tanh','relu'],[0.00005,0.0001,0.001],['constant','invscaling','adaptative']]
    best_mlp = []
    #ENTRENAMIENTO DE ALGORITMOS
    for alg in modelos:
        if alg == 'Svc':
            for a in params_svc[0]:
                for b in params_svc[1]:
                    for c in params_svc[2]:
                        for d in params_svc[3]:
                            for e in params_svc[4]:
                                mod = SVC(kernel=a,degree=b,coef0=c,probability=d,shrinking=e)
                                punt = entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
                                if punt > puntuaciones[0]:
                                    puntuaciones[0] = punt
                                    best_svc = [a,b,c,d,e]
        elif alg == 'NaiveBayes':
            mod = MultinomialNB()
            puntuaciones[1] = entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
        elif alg == 'Knn':
            for a in params_knn[0]:
                for b in params_knn[1]:
                    for c in params_knn[2]:
                        for d in params_knn[3]:
                            for e in params_knn[4]:
                                mod = KNeighborsClassifier(n_neighbors=a,weights=b,algorithm=c,leaf_size=d,p=e)
                                punt = entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
                                if punt > puntuaciones[2]:
                                    puntuaciones[2] = punt
                                    best_knn = [a,b,c,d,e]
        elif alg == 'Mlp':
            for a in params_mlp[0]:
                for b in params_mlp[1]:
                    for c in params_mlp[2]:
                        for d in params_mlp[3]:
                            mod = MLPClassifier(hidden_layer_sizes=a,activation=b,alpha=c,learning_rate=d)
                            punt = entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod)
                            if punt > puntuaciones[3]:
                                puntuaciones[3] = punt
                                best_mlp = [a,b,c,d]
    print(puntuaciones)
    tmp = 0
    guia = 0
    for h in puntuaciones:
        if h > tmp:
            best_model = guia
            tmp = h
        guia = guia + 1
    entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, SVC(kernel=best_svc[0],degree=best_svc[1],coef0=best_svc[2],probability=best_svc[3],shrinking=best_svc[4]))
    entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod = MultinomialNB())
    entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, KNeighborsClassifier(n_neighbors=best_knn[0],weights=best_knn[1],algorithm=best_knn[2],leaf_size=best_knn[3],p=best_knn[4]))
    entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, MLPClassifier(hidden_layer_sizes=best_mlp[0],activation=best_mlp[1],alpha=best_mlp[2],learning_rate=best_mlp[3]))

def entrenar(alg,train_vectors,train_labels,test_vectors, test_labels, mod):
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
    #print("MODELO ", alg, " ENTRENADO Y PROBADO")
    #print(classification_report(test_labels, mod.predict(test_vectors)))
    #print(confusion_matrix(test_labels, mod.predict(test_vectors)))
    predicted = cross_val_predict(mod, test_vectors, test_labels, cv=10)
    print("Cross validation %s" % accuracy_score(test_labels, predicted))
    return accuracy_score(test_labels,predicted)

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
        df3 =pd.DataFrame(data=dataset['text'])
        dfNER = pd.DataFrame(data=dataset['text'])
        dfNER = tokenizar(dfNER)
        anNER = dfNER['final'][-5:]
        resultadoNER = usar_NER(anNER,3)
        for ent in resultadoNER:
            for it in ent:
                print(it)
                df3 = df3[df3['text'].str.contains(it[0])]
                df3 = tokenizar(df3)
                test_data = df3['final'][-100:] #saca sólo los últimos 100
                test_data = list(test_data.apply(' '.join))
                test_vectors = vectorizer.transform(test_data)
                mostrar_graph(predecir_Naive_Bayes(test_vectors, it),predecir_SVC(test_vectors, it), predecir_KNN(test_vectors, it), predecir_MLP(test_vectors, it))
                df3 =pd.DataFrame(data=dataset['text'])#vuelve a recargar el df3
'''
Funciones de predicción de los diferentes algoritmos para los diferentes modelos
'''
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
'''
Función que utiliza NER para detectar entidades.
'''
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
'''
Función que analiza un único tweet y muestra la información.
'''
def analizar_tweet(consulta, iteraciones, tweet_por_iteracion, vectorizer):
    #Claves y tokens de la cuenta de twitter
    consumer_key='ynSB0dFvqPl3xRU7AmYk39rGT'
    consumer_secret='6alIXTKSxf0RE57QK3fDQ8dxdvlsVr1IRsHDZmoSlMx96YKBFD'
    access_token='966591013182722049-BVXW14Hf5s6O2oIwS3vtJ3S3dOsKLbY'
    access_token_secret='829DTKPjmwsSytmp1ky9fMCJkjV0LZ04TbL9oqHGV6cDm'
    #parámetros de la consulta
    q = 'premier league -filter:retweets AND -filter:replies'
    url = 'https://api.Twitter.com/1.1/search/tweets.json'
    pms = {'q' : q, 'count' : 10, 'lang' : 'en', 'result_type': 'recent'} 
    auth = OAuth1(consumer_key, consumer_secret, access_token,access_token_secret)
    #inicialización de la base de datos para cargar los datos
    database_name = "baseDeDatos"
    collection_name = "coleccion"
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]
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
    #Pasar de la base de datos a un dataframe
    ##################################################################
    documents = []
    for doc in collection.find():
        documents.append(doc)
    df = pd.DataFrame(documents)
    mostrar = pd.DataFrame(documents)
    #Limpieza de datos
    df = limpieza_de_datos_de_twitter(df)
    df2 = pd.DataFrame(data=df['text'][-1:])
    dfNER = pd.DataFrame(data=df['text'][-1:])
    tweet = mostrar['text'][-1:]
    dfNER = tokenizar(dfNER)
    anNER = dfNER['final'][-1:]
    print(anNER)
    resultadoNER = usar_NER(anNER,10)

    df2 = tokenizar(df2)
    test_data = df2['final']
    test_data = list(test_data.apply(' '.join))
    test_vectors = vectorizer.transform(test_data)

    nb = predecir_Naive_Bayes(test_vectors, 'nada')
    svc = predecir_SVC(test_vectors, 'nada')
    knn = predecir_KNN(test_vectors, 'nada')
    mlp = predecir_MLP(test_vectors, 'nada')

    print("\tTweet: ", tweet)
    print("\tEntidades: ")
    if resultadoNER == 0:
        print("\tNo se reconoció ninguna entidad")
    else:
        for i in resultadoNER:
            for j in i:
                print("\t - ", j[0])
    print("\tSentimiento con Naive Bayes")
    if nb[1][0] == 1:
        print("\t\tPositivo")
    elif nb[1][1] == 1:
        print("\t\tNeutro")
    elif nb[1][2] == 1:
        print("\t\tNegativo")
    print("\tSentimiento con Clasificador SVC")
    if svc[1][0] == 1:
        print("\t\tPositivo")
    elif svc[1][1] == 1:
        print("\t\tNeutro")
    elif svc[1][2] == 1:
        print("\t\tNegativo")
    print("\tSentimiento con Clasificador K-Neighbors")
    if knn[1][0] == 1:
        print("\t\tPositivo")
    elif knn[1][1] == 1:
        print("\t\tNeutro")
    elif knn[1][2] == 1:
        print("\t\tNegativo")
    print("\tSentimiento con Clasificador MLP")
    if mlp[1][0] == 1:
        print("\t\tPositivo")
    elif mlp[1][1] == 1:
        print("\t\tNeutro")
    elif mlp[1][2] == 1:
        print("\t\tNegativo")
'''
Función del programa principal
'''
def programa_principal(vectorizer):
    choice = mostrar_menu_principal()
    if choice == '1':
        print("\nAnalizar un bloque de tweets\n")
        analizar_bloque_de_tweets(vectorizer)
    elif choice == '2':
        print("\nAnalizar un tweet individual\n")
        analizar_tweet('premier league -filter:retweets AND -filter:replies',1,1,vectorizer)
    elif choice == '3':
        os.system('cls') #limpiar la pantalla del terminal
        print("\nEntrenar algoritmos\n")
        entrenar_algoritmos(vectorizer)
    elif choice == 'q':
        print("\nAdios!")
    else:
        print("\nSelecciona una opción correcta\n")
        
programa_principal(vectorizer)