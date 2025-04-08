import pixeltable as pxt
import pandas as pd
import time

# Conectar a la tabla existente de consultas
print("Conectando a la tabla de consultas...")
queries_data = pxt.get_table('rag_demo.queries')

# Agregar una nueva consulta
print("Agregando nueva consulta sobre métodos numéricos...")
queries_data.insert([{
    'question': '¿Qué son los métodos numéricos y cuál es su importancia en la mecánica computacional?',
    'answer': 'Los métodos numéricos son técnicas para resolver problemas matemáticos complejos mediante aproximaciones numéricas. Son fundamentales en la mecánica computacional para resolver ecuaciones diferenciales que no tienen solución analítica.'
}])

# Esperar a que los resultados de la consulta se procesen
print("Esperando a que se procese la nueva consulta...")
# Pequeña pausa para asegurar que el pipeline RAG tenga tiempo de procesar
time.sleep(5)
# Recuperar todos los resultados, incluyendo la nueva consulta, usando pandas
print("Recuperando resultados...")
resultados_df = queries_data.select(
    queries_data.question,
    queries_data.answer,
    queries_data.response
).collect()

# Convertir a DataFrame de pandas si no lo es ya
if not isinstance(resultados_df, pd.DataFrame):
    resultados_df = pd.DataFrame(resultados_df)

# Exportar los resultados a un archivo CSV usando pandas
ruta_exportacion = r'c:\Users\Kieff\Desktop\Pixeltable\RAG-PPS-Pixeltable\resultados_actualizados.csv'
resultados_df.to_csv(ruta_exportacion, index=False, encoding='utf-8')

print(f"Resultados exportados exitosamente a: {ruta_exportacion}")
print(f"Total de consultas en la tabla: {len(resultados_df)}")

# Mostrar la nueva consulta y su respuesta usando pandas
print("\nNueva consulta agregada:")
nueva_consulta_df = queries_data.where(
    queries_data.question == '¿Qué son los métodos numéricos y cuál es su importancia en la mecánica computacional?'
).select(
    queries_data.question,
    queries_data.answer,
    queries_data.response
).collect()

# Si hay resultados, mostrarlos
if not nueva_consulta_df.empty:
    print(f"Pregunta: {nueva_consulta_df.iloc[0]['question']}")
    print(f"Respuesta esperada: {nueva_consulta_df.iloc[0]['answer']}")
    print(f"Respuesta del modelo: {nueva_consulta_df.iloc[0]['response']}")