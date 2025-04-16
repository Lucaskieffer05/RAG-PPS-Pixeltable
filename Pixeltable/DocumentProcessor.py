import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import clip
from pixeltable.functions.ollama import chat

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
        
        self.queries_table = pxt.create_table(
            f'{self.directory}.queries',
            {
                'question': pxt.String,
            }
        )

        self.documents_table = pxt.create_table(
            f'{self.directory}.documents',
            {'document': pxt.Document}
        )

        self.chunks_view = pxt.create_view(
            f'{self.directory}.chunks',
            self.documents_table,
            iterator=DocumentSplitter.create(
                document=self.documents_table.document,
                separators='token_limit',
                limit=chunkLimit
            )
        )

        self.chunks_view.add_embedding_index(
            'text',
            embedding=clip.using(model_id=self.embed_model)
        )

        self.queries_table.add_computed_column(
            question_context=self._get_top_chunks(self.chunks_view,self.queries_table.question)
        )
        
        self.queries_table.add_computed_column(
            prompt=self._create_prompt(self.queries_table.question_context, self.queries_table.question)
        )

        self.queries_table.add_computed_column(raw_output=chat(
            messages=self._create_messages(self.queries_table.prompt),
            model=self.inference_model,
            options={'max_tokens': max_tokens, 'top_p': top_p, 'temperature': temperature},
        ))

        self.queries_table.add_computed_column(output_content=self.queries_table.raw_output.message.content)

    
    # Obtiene los mejores chunks para responder una pregunta

    def _get_top_chunks(chunks_view: pxt.View, query_text: str, kSize: int = 5):
        sim = chunks_view.text.similarity(query_text)
        return (
            chunks_view.order_by(sim, asc=False)
                .select(chunks_view.text, sim=sim)
                .limit(kSize)
        )

    @staticmethod
    @pxt.udf
    def _create_prompt(top_k_list: list[dict], question: str) -> str:
        concat_top_k = '\n\n'.join(
            elt['text'] for elt in reversed(top_k_list)
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

    # 
    def insert_documents(self, 
                         namesDocuments : list,
                         basePathDocuments: str = r'Pixeltable/Documents/Mecanica Computacional/'
                         ) -> None:
        documents_table = pxt.get_table(f'{self.directory}.documents')
        for name in namesDocuments:
            documents_table.insert([{'document': basePathDocuments + name}])
        
    def get_answer (self, question: str):        
        try:
            queries_table = pxt.get_table(f'{self.directory}.queries')
            response = queries_table.insert([{'question': question}])
            queries_table.delete()
            return response[0]    
        except Exception as e:
            return f"Error al insertar la pregunta: {str(e)}"

        