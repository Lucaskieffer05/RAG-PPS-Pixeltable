import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import clip
from pixeltable.functions.ollama import chat
import pixeltable.exceptions as excs # Importar excepciones
import inspect # Para inspección de errores
from pixeltable.functions.huggingface import sentence_transformer
from datetime import datetime

import logging
import sys
import os

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Configuración de encoding y logging (compatible con Jupyter)
os.environ["PYTHONUTF8"] = "1"

# Configurar logging para UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if isinstance(msg, str):
                msg = msg.encode('utf-8', 'replace').decode('utf-8')
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    handlers=[UTF8StreamHandler(sys.stdout)],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Queries: Tabla donde se guardan las preguntas y respuestas
# Documents: Tabla donde se guardan los documentos
class DocumentProcessor:

    def __init__(self, directory: str = "rag_documents", 
                 inference_model: str = "llama3.2:3b",
                 embded_model: str = "openai/clip-vit-base-patch32"):
        self.directory = directory
        self.embed_model = embded_model
        self.inference_model = inference_model
    
    # Crea tablas de queries, documentos y views de chunks (contiene embeddings) [Se debe correr una sola vez]
    def setup(self, chunkLimit: int = 512, max_tokens : int = 4096, top_p: float = 0.9, temperature: float = 0.5) -> None:
        
        # Si ya existe la tabla, no la vuelve a crear
        self.queries_table = pxt.create_table(
            f'{self.directory}.queries',
            {
                'question': pxt.String,
                'created_at': pxt.Timestamp,
            }
        )

        self.documents_table = pxt.create_table(
            f'{self.directory}.documents',
            {'document': pxt.Document}
        )

        self.insert_documents(['Leyes2024.pdf'])

        self.chunks_view = pxt.create_view(
            f'{self.directory}.chunks',
            self.documents_table,
            iterator=DocumentSplitter.create(
                document=self.documents_table.document,
                separators='paragraph, token_limit',
                limit=chunkLimit,
                metadata = 'page'
            )
        )

        print("Realizando particiones del documento.")

        self.chunks_view.add_embedding_index(
            'text',
            embedding=sentence_transformer.using(model_id='intfloat/e5-large-v2')
        )
        

        self.chunks_view.add_computed_column(adjusted_page=self.adjust_page_num(self.chunks_view.page))

        @pxt.query
        def _get_top_chunks(query_text : str, kSize: int = 10):
            sim = self.chunks_view.text.similarity(query_text)
            return (
                self.chunks_view.order_by(sim, asc=False)
                    .select(self.chunks_view.adjusted_page, self.chunks_view.text, sim=sim)
                    .limit(kSize)
            )
        
        self.queries_table.add_computed_column(
            question_context=_get_top_chunks(self.queries_table.question)
        )

        print("Tomando el top k y dando contexto.")
        
        self.queries_table.add_computed_column(
            prompt=self._create_prompt(self.queries_table.question_context, self.queries_table.question)
        )

        print("Creando el prompt.")
        self.queries_table.add_computed_column(raw_output=chat(
            messages=self._create_messages(self.queries_table.prompt),
            model=self.inference_model,
            options={'max_tokens': max_tokens, 'top_p': top_p, 'temperature': temperature},
        ))

        print("Generando la respuesta.")
        self.queries_table.add_computed_column(output_content=self.queries_table.raw_output.message.content)

        #print(self.chunks_view.select(self.chunks_view.adjusted_page, self.chunks_view.text).collect())
    
    # Obtiene los mejores chunks para responder una pregunta

    @staticmethod
    @pxt.udf
    def _create_prompt(top_k_list: list[dict], question: str) -> str:
        concat_top_k = '\n\n'.join(
            f"Page: {elt['adjusted_page']}\n{elt['text']}" for elt in reversed(top_k_list)
        )
        return f'''
        PASSAGES:

        {concat_top_k}

        QUESTION:

        {question}'''

    @staticmethod
    @pxt.udf
    def _create_messages( prompt: str) -> list[dict]:
        return [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

    @pxt.udf
    def adjust_page_num(page: int) -> int:
        return page + 1  # Suma 1 a cada número de página
    
    def insert_documents(self, 
                         namesDocuments : list,
                         basePathDocuments: str = r'Documents/Leyes/'
                         ) -> None:
        documents_table = pxt.get_table(f'{self.directory}.documents')
        for name in namesDocuments:
            documents_table.insert([{'document': basePathDocuments + name}])
        
    def get_answer (self, question: str):        
        try:
            queries_table = pxt.get_table(f'{self.directory}.queries')
            now = datetime.now()
            queries_table.insert([{'question': question, 'created_at': now}])
            print("Pregunta insertada.")
            #print(queries_table.select(queries_table.prompt).collect())
            return queries_table.select(queries_table.output_content).order_by(queries_table.created_at, asc=False).limit(1).collect()
        except Exception as e:
            return f"Error al insertar la pregunta: {str(e)}"

        