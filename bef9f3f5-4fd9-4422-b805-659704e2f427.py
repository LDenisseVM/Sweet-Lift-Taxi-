#!/usr/bin/env python
# coding: utf-8

# # Hola Denisse!
# 
# Mi nombre es David Bautista, soy code reviewer de Tripleten y hoy tengo el gusto de revisar tu proyecto.
# 
# Cuando vea un error laa primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión.
# 
# Encontrarás mis comentarios más abajo - por favor, no los muevas, no los modifiques ni los borres.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div>
# 
# ¡Empecemos!

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# # Comentario General
#     
# Hola Denisse, te felicito por el desarrollo del proyecto, realizaste de manera correcta las diferentes secciones. </div>

# # Descripción del proyecto
# 
# La compañía Sweet Lift Taxi ha recopilado datos históricos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. Construye un modelo para dicha predicción.
# 
# La métrica RECM en el conjunto de prueba no debe ser superior a 48.
# 
# ## Instrucciones del proyecto.
# 
# 1. Descarga los datos y haz el remuestreo por una hora.
# 2. Analiza los datos
# 3. Entrena diferentes modelos con diferentes hiperparámetros. La muestra de prueba debe ser el 10% del conjunto de datos inicial.4. Prueba los datos usando la muestra de prueba y proporciona una conclusión.
# 
# ## Descripción de los datos
# 
# Los datos se almacenan en el archivo `taxi.csv`. 	
# El número de pedidos está en la columna `num_orders`.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con la sección de introducción del proyecto.
#  </div>

# ## Preparación

# In[86]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo importando las librerías necesarias para el desarrollo del proyecto.</div>

# In[87]:


data= pd.read_csv('/datasets/taxi.csv', parse_dates=[0], index_col=[0])


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con la carga del dataset.</div> 

# In[88]:


data.head(10)


# In[89]:


data.info()


# In[90]:


data.describe()


# Haremos el remuestreo por hora:

# In[91]:


data=data.resample('1H').sum()
data


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy buen trabajo con esta exploración inicial de los datos, Denisee. Utilizas componentes útiles e importantes de análisis.  </div>

# ## Análisis

# In[92]:


data.plot()


# Vemos que hay mas pedidos de taxis en verano, sin embargo lo que nos interesan son las horas pico de un día, asi que graficaremos un segmento más pequeño, de un día

# In[93]:


data['2018-03-01'].plot()


# Vemos que el número más grande de pedidos es alrededor de la media noche, mientras que cuando hay menos demanda es alrededor de las 6:00 am 

# Ahora analicemos una semana y un mes distintos

# In[94]:


data['2018-03-01': '2018-03-07'].plot()


# In[95]:


data['2018-04-01': '2018-04-07'].plot()


# Vemos que hay mayor demanda los lunes y sabados en promedio

# In[96]:


data['2018-03'].plot()


# In[97]:


data['2018-04'].plot()


# Vemos que alrededor del día 23 de cada mes hay una mayor demanda en los pedidos de taxi

# Crearemos un modelo basado en el mes, dia y hora 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo, Denisse. Usas secciones interesantes e importantes en el desarrollo de esta sección de análisis de componentes propios de la serie de tiempo.  </div>

# ## Formación

# In[98]:


data['month'] = data.index.month
data['day'] = data.index.day
data['dayofweek'] = data.index.dayofweek
data['hour']= data.index.hour


# In[99]:


train, test= train_test_split(data, test_size=.1, shuffle= False)


# In[100]:


features_train=train.drop(['num_orders'], axis=1)
target_train= train['num_orders']
features_test=test.drop(['num_orders'], axis=1)
target_test= test['num_orders']


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo creando los sets de entreanmiento y testeo y asi mismo separando los datos en caracteristicas y targets.  </div>

# ## Prueba

# In[101]:


model_rf= RandomForestRegressor(n_estimators= 100, max_depth=10)
model_rf.fit(features_train, target_train)
predictions_rf= model_rf.predict(features_test)
RECM= mean_squared_error(target_test, predictions_rf, squared=False)
RECM


# In[102]:


model_dt= DecisionTreeRegressor(max_depth=500)
model_dt.fit(features_train, target_train)
predictions_dt= model_dt.predict(features_test)
RECM= mean_squared_error(target_test, predictions_dt, squared=False)
RECM


# In[103]:


model_lr= LinearRegression()
model_lr.fit(features_train, target_train)
predictions_lr= model_lr.predict(features_test)
RECM= mean_squared_error(target_test, predictions_lr, squared=False)
RECM


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo con el desarrollo y las pruebas de los resultados de los modelos.  </div>

# Vemos que con las caracteristicas que creamos logramos que el RECM del modelo sea menor a 48 (Bosque aleatorio y árbol de decisión)

# # Lista de revisión

# - [x]  	
# Jupyter Notebook está abierto.
# - [x]  El código no tiene errores
# - [x]  Las celdas con el código han sido colocadas en el orden de ejecución.
# - [x]  	
# Los datos han sido descargados y preparados.
# - [x]  Se ha realizado el paso 2: los datos han sido analizados
# - [x]  Se entrenó el modelo y se seleccionaron los hiperparámetros
# - [x]  Se han evaluado los modelos. Se expuso una conclusión
# - [x] La *RECM* para el conjunto de prueba no es más de 48
