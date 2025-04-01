import numpy as np
import pixeltable as pxt

# Ensure a clean slate for the demo
pxt.drop_dir('rag_demo', force=True)
pxt.create_dir('rag_demo')

# ---------------------------------------------------------------------------------------------------
# Cargar preguntas desde un archivo JSON
# ---------------------------------------------------------------------------------------------------

# Corregir las rutas usando cadenas raw o barras normales
base = r'Pixeltable/Documents/Mecanica Computacional'
# Cargar preguntas desde un archivo JSON
json_url = base + '/preguntas.json'

# Importar los datos desde el archivo JSON y permitir que cree la tabla automáticamente
queries_data = pxt.io.import_json(
    filepath_or_url=json_url,
    tbl_path='rag_demo.queries'  # Especificar el nombre de la tabla
)

print("Datos importados correctamente.")

# ---------------------------------------------------------------------------------------------------
# Cargar documentos
# ---------------------------------------------------------------------------------------------------

documents_t = pxt.create_table(
    'rag_demo.documents',
    {'document': pxt.Document}
)

documents_t.insert(
    [{'document': base + '/Apuntes_Meco.pdf'}]
)

print("Documentos cargados correctamente.")


# ---------------------------------------------------------------------------------------------------
# Realizar particiones del documento
# ---------------------------------------------------------------------------------------------------

from pixeltable.iterators import DocumentSplitter

chunks_t = pxt.create_view(
    'rag_demo.chunks',
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.document,
        separators='token_limit',
        limit=300
    )
)


# ---------------------------------------------------------------------------------------------------
# incrustal indexar documentos
# ---------------------------------------------------------------------------------------------------

from pixeltable.functions.huggingface import sentence_transformer

chunks_t.add_embedding_index(
    'text',
    embedding=sentence_transformer.using(model_id='intfloat/e5-large-v2')
)

query_text = "¿Que es Diferencias Finitas (FDM)?"
sim = chunks_t.text.similarity(query_text)
nvidia_eps_query = (
    chunks_t
    .order_by(sim, asc=False)
    .select(similarity=sim, text=chunks_t.text)
    .limit(5)
)

# ---------------------------------------------------------------------------------------------------
# Tomamos un top k y damos contexto
# ---------------------------------------------------------------------------------------------------


@pxt.query
def top_k(query_text: str):
    sim = chunks_t.text.similarity(query_text)
    return (
        chunks_t.order_by(sim, asc=False)
            .select(chunks_t.text, sim=sim)
            .limit(5)
    )

queries_data.add_computed_column(
    question_context=top_k(queries_data.question)
)

# ---------------------------------------------------------------------------------------------------
# Creamos la prompt para el modelo LLM
# ---------------------------------------------------------------------------------------------------


@pxt.udf
def create_prompt(top_k_list: list[dict], question: str) -> str:
    concat_top_k = '\n\n'.join(
        elt['text'] for elt in reversed(top_k_list)
    )
    return f'''
    PASSAGES:

    {concat_top_k}

    QUESTION:

    {question}'''

queries_data.add_computed_column(
    prompt=create_prompt(queries_data.question_context, queries_data.question)
)

# ---------------------------------------------------------------------------------------------------
# Hacemos las preguntas
# ---------------------------------------------------------------------------------------------------
from pixeltable.functions import llama_cpp

@pxt.udf
def create_messages(prompt: str) -> list[dict]:
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt},
    ]

queries_data.add_computed_column(result=llama_cpp.create_chat_completion(
    messages=create_messages(queries_data.prompt),
    repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    repo_filename='*q5_k_m.gguf'
))


queries_data.add_computed_column(output=queries_data.result.choices[0].message.content)

queries_data.select(queries_data.question, queries_data.answer).collect()