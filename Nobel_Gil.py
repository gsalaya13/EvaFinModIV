import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.write(''' # Predicción de categoría en Premios Nobel. ¡Dame tu discurso en inglés! ''')
st.image("premios_nobel.png", caption="Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895. Es el premio internacional más importante que existe en el mundo de la cultura y científicos y que se otorga cada año para reconocer a personas o instituciones que hayan llevado a cabo investigaciones, descubrimientos o contribuciones notables a la sociedad en diversas áreas.")

#El siguiente es encabezado de sección
st.header('Texto')

#######################################
#Aquí se captura el input del usuario de la página
def user_input_features():
  #Entrada
  texto = st.text_input("Introduce el texto a evaluar en inglés")

  user_input_data = {'Text': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features
  #Termina la función
######################################

#Se guarda el input del usuario en un dataframe llamado df
df = user_input_features()

#Aquí se cargan los datos de entrenamiento, el archivo que exporté definitivo
#de solo dos columnas y el texto ya procesado y categorías codificadas
nobel =  pd.read_csv('df_def_nobel.csv', encoding='utf-8')
X = nobel.Text_proc
y = nobel.Label

#Vectorización
#Convirtiendo el texto en matriz de conteos
#Aprender vocabulario de X y transformar textos en vectores
vect = CountVectorizer()
#X_dtm es la matriz lista para el modelo
X_dtm = vect.fit_transform(X)

#Creando el clasificador a ocupar
#Con esto entreno al modelo con el texto vectorizado y con
#las etiquetas de cada categoría, es decir Label
nb = MultinomialNB()
nb.fit(X_dtm, y)

#Transformando el texto del input del usuario al mismo formato de matriz que 
#ocupe en los datos de entrenamiento para poder ser evaluado
#Me devuelve la predicción
df_dtm = vect.transform(df['Text'])
prediction = nb.predict(df_dtm)
#prediction = nb.predict(df_dtm)[0]

#{'chemistry':0,'economics':1,'literature':2,'medicine':3,'peace':4,'physics':5}
#'Chemistry','Economics','Literature','Medicine','Peace','Physics'
#Se pudo poner así también
#labels = ['chemistry':0,'economics':1,'literature':2,'medicine':3,'peace':4,'physics':5]
#st.write(labels[prediction[0]])
st.subheader('Predicción')
if prediction == 0:
  st.write('Química')
elif prediction == 1:
  st.write('Economía')
elif prediction == 2:
  st.write('Literatura')
elif prediction == 3:
  st.write('Medicina')
elif prediction == 4:
  st.write('Paz')
elif prediction == 5:
  st.write('Física')
else:
  st.write('Sin predicción')
