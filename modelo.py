import pandas as pd

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

