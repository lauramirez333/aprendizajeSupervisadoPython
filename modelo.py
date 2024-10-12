import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Modelo supervisado
from sklearn.metrics import mean_squared_error      # Para evaluar el modelo


#Creamos las fuentes de datos para el aprendizaje supervisado
#Dataframe de Estaciones y Rutas
estaciones = {'id_estacion_origen': ['A', 'A', 'B', 'D', 'C'],
        'id_estacion_destino': ['B','D','C','C','E',],
        'tiempo_viaje': [10,20,15,5,10],
        'frecuencia_viaje': [6,4,5,7,6],
        'congestion': ['bajo', 'medio', 'alto', 'bajo','medio']}

dfEstaciones = pd.DataFrame(estaciones)
dfEstaciones

#Dataset de Demanda de Pasajeros
demandaPasajeros = {'id_estacion': ['A', 'B', 'C', 'D', 'E'],
        'hora_dia': [7,7,17,12,19],
        'pasajeros': [100, 200, 50, 80, 120],
        'dia_semana': ['Lunes','Lunes', 'Viernes', 'Martes', 'Miercoles']}

dfDemandaPasajeros = pd.DataFrame(demandaPasajeros)
dfDemandaPasajeros

#Dataset de Incidentes/Mantenimiento
incidentes = {'id_estacion': ['C', 'D'],
        'tipo_incidente': ['Mantenimiento', 'Cierre'],
        'fecha_inicio': ['2023-10-01', '2023-10-05'],
        'fecha_fin': ['2023-10-05','2023-10-06']}

dfIncidentes = pd.DataFrame(incidentes)
dfIncidentes

#Dataset de Clima
clima = {'id_estacion': ['A'],
        'fecha': ['2023-10-05'],
        'clima': ['Lluvia']}

dfClima = pd.DataFrame(clima)
dfClima

#Unimos los datos usando merge
df_rutas_demanda = pd.merge(dfEstaciones, dfDemandaPasajeros, left_on='id_estacion_origen', right_on='id_estacion', how='left')
df_rutas_demanda

#Unimos el los datos con los del clima 
df_rutas_clima = pd.merge(df_rutas_demanda, dfClima, left_on=['id_estacion_origen'], right_on=['id_estacion'], how='left')
df_rutas_clima

# Filtrar los incidentes que afecten la ruta (si están dentro del rango de fechas)
def incidente_activo(fecha, estacion):
    incidentes = dfIncidentes[(dfIncidentes['id_estacion'] == estacion) &
                               (dfIncidentes['fecha_inicio'] <= fecha) &
                               (dfIncidentes['fecha_fin'] >= fecha)]
    return 1 if not incidentes.empty else 0

# Agregar columna de incidente activo
df_rutas_clima['incidente_activo'] = df_rutas_clima.apply(lambda row: incidente_activo(row['fecha'], row['id_estacion_origen']), axis=1)

# Visualizamos el dataset combinado
print(df_rutas_clima.head())

#Entrenamos un modelo supervisado
# Variables predictoras (features)
X = df_rutas_clima[['tiempo_viaje', 'pasajeros', 'incidente_activo']]

# Variable objetivo (target): Tiempo ajustado que queremos predecir
# Si quieres predecir 'tiempo_viaje', asegúrate de que sea la columna correcta
y = df_rutas_clima['tiempo_viaje']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest Regressor
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
modelo_rf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = modelo_rf.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f"Error Cuadrático Medio (MSE): {mse}")

def predecir_tiempo_ajustado(row):
    # Usar el modelo para predecir el tiempo ajustado
    tiempo_ajustado = modelo_rf.predict([[row['tiempo_viaje'], row['pasajeros'], row['incidente_activo']]])
    return tiempo_ajustado[0]

# Aplicar el modelo en el conjunto de rutas para obtener tiempos ajustados
df_rutas_clima['tiempo_ajustado'] = df_rutas_clima.apply(predecir_tiempo_ajustado, axis=1)

# Visualizamos los tiempos ajustados
print(df_rutas_clima[['id_estacion_origen', 'id_estacion_destino', 'tiempo_viaje', 'tiempo_ajustado']])
