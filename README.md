# Twitter Sentiments
Este repositorio pertenece al Proyecto de fin de grado de Ingeniería informática.
El objetivo de dicho proyecto es el siguiente:
Comprender y evaluar el funcionamiento de las técnicas de análisis de
sentimiento y reconocimiento de entidades, así como desarrollar un aplicativo que
implemente dichas técnicas y muestre los resultados obtenidos. Concretamente se
pretende desarrollar un aplicativo utilizando Qt for Python mediante el cual se pueda
obtener datos de Twitter.com o desde un archivo para entrenar cuatro tipos distintos
de algoritmos y una vez entrenados se pueda obtener información acerca de las
entidades y sentimientos que se desprenden de cualquier tweet o bloque de tweets
que se desee analizar.
## Más información
Toda la información referente al proceso de desarrollo y todas las notas relacionadas en cuanto al análisis de sentimientos, se pueden encontrar en el siguiente blog: https://tfgatusaer.blogspot.com.es/
## Ficheros
| Ficheros | Descripción |
| ---------- | ---------- |
| Bloque de Tweets   |  Fichero csv que se utiliza para cargar los datos desde un archivo  |
| Entrenamiento 1 y 2   |  Fichero csv para cargar los datos de entrenamiento  |
| Resto | Resto de tweets sin etiquetar   |
| Knn, Mlp, NaiveBayes, Svc  | Ficheros donde se almacenan los modelos entrenados   |
| fvecto   | Fichero donde se almacena el vector de los modelos   |
| PFG_REST_API   | Código de REST API con el funcionamiento básico   |
| PFG_Streaming_API   | Código de Streaming API con el funcionamiento básico   |
| script   | Código de funcionamiento completo para terminal   |
| qt_script   | Código de funcionamiento completo con interfaz de usuario usando qt   |
