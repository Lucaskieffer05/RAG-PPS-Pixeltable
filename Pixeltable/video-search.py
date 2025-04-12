import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.video import extract_audio
from pixeltable.iterators.string import StringSplitter
from openai import vision
from pixeltable.ext.functions.yolox import yolox
import PIL.Image
import PIL.ImageDraw
import pytesseract # OCR
from PIL import ImageOps, ImageEnhance
from typing import Literal

EMBED_MODEL = sentence_transformer.using(model_id='intfloat/e5-large-v2')
DIRECTORY = 'video_detection'
TABLE_NAME = 'video_detection.videos'
EXTRACTED_FRAMES_PER_SECOND = 1 
VIDEO_SEARCH_RESULTS = 5

#############################################################################################
#                                   Etapa de configuración                                        
#############################################################################################
print("Empezando configuración...")
def initialize_pixeltable(dir_name=DIRECTORY):
    pxt.drop_dir(dir_name, force=True)
    pxt.create_dir(dir_name)
    
initialize_pixeltable()

videos = pxt.create_table(
  DIRECTORY,
  {'video': pxt.VideoType()},
  'uploaded_at': pxt.Timestamp,
  if_exists='ignore'
)

videos.add_computed_column(audio_extract=extract_audio(videos.video, 
                                                       format='mp3')) 

# View para videos
frames = pxt.create_view(
    f'{DIRECTORY}.frames',
    videos,
    iterator=pxt.iterators.FrameIterator.create(
        video=videos.video,
        fps=EXTRACTED_FRAMES_PER_SECOND
    )
)
 
frames.add_computed_column(
    image_description=vision(
        prompt="Provide quick caption for the image.",
        image=frames.frame,
        model="gpt-4o-mini"
    )
)  

frames.add_embedding_index('image_description', 
                           string_embed=EMBED_MODEL)

# View para audio
chunks_view = pxt.create_view(
    f'{DIRECTORY}.video_chunks',
    videos,
    iterator=pxt.iterators.AudioChunkIterator.create(
        audio=videos.audio_extract,
        chunk_duration_sec=10.0,
        overlap_sec=2.0,
        min_chunk_duration_sec=5.0
    )
)

# Audio a texto
chunks_view.add_computed_column(
    text=pxt.functions.audio_to_text(chunks_view.audio, language='es', model='whisper-1')
)

# View que convierte texto a sentencias
transcription_chunks = pxt.create_view(
    f'{DIRECTORY}.video_sentence_chunks',
    chunks_view,
    iterator=StringSplitter.create(text=chunks_view.transcription.text, separators='sentence'),
)

# Embedding para audio
transcription_chunks.add_embedding_index('text', string_embed=EMBED_MODEL)

# def process_video(filePath: str):
#   try:
#     videos = pxt.get_table('detection.videos')
#     videos.insert([{'video': str(filePath)}])
    
#     print(f"Video procesado {filePath}...")
#   except Exception as e:
#     print(f"Error procesando video: {filePath}. Error: {e}")
    
  
# def search_video(query, 
#                  search_type: Literal['text', 'image'], 
#                  num_results: int):
#   try: 
#     frames = pxt.get_table('videos.frames')
    
#     if search_type == 'text':
#       text_query = query.
#       sim = frames.frame.similarity(text_query)
#     else: # es imagen
#       if not query:
#         raise ValueError("La query no puede estar vacía")
#       image = PIL.Image.open()
      
#       sim = frames_view.frame.similarity(image)
    
#     results = frames.odery_by(sim, asc=False).limit(num_results).select(frames.frame).collect()
    
