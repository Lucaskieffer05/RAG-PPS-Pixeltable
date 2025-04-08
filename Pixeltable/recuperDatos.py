import pixeltable as pxt
import pandas as pd

pd.set_option('display.max_colwidth', None)


print("Recuperando resultados de consultas anteriores...")
queries_data = pxt.get_table('rag_demo.queries')

print(queries_data.head(5))
