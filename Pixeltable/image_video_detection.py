import pixeltable as pxt
from pixeltable.iterators import FrameIterator
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.video import extract_audio

# Esto chequear después (tema de embeddings)
from pixeltable.functions.huggingface import sentence_transformer, clip
from pixeltable.ext.functions.yolox import yolox
import PIL.Image
import pytesseract  
from PIL import ImageOps, ImageEnhance

def initialize_pixeltable(dir_name='detection'):
    pxt.drop_dir(dir_name, force=True)
    pxt.create_dir(dir_name)

# class ImageProcessor:
#     def __init__(self, directory : str = "detection", model : str ='yolox_s', confidence_threshold : float = 0.5):
#         self.directory = directory
#         self.model = model
#         self.confidence_threshold = confidence_threshold

#     def setup(self) -> None:
#         images = pxt.create_table(self.directory, {'image': pxt.ImageType()}, if_exists='ignore')
#         self.images_table = images
    
    
#     # Método para buscar imágenes similares por texto o imagen
#     def search(self, search_type, text_query=None, image_query=None, limit=5):
#         try:
#             if search_type == "Text" and text_query:
#                 sim = self.table.image.similarity(text_query)
#             elif search_type == "Image" and image_query is not None:
#                 sim = self.table.image.similarity(image_query)
#             else:
#                 return []
                
#             results = self.table.order_by(sim, asc=False).limit(limit).select(
#                 self.table.image, 
#                 self.table.detections, 
#                 getattr(self.table, 'license_plate_text', None)
#             ).collect()
#             return results
#         except Exception as e:
#             print(f"Error en la búsqueda: {str(e)}")
#             return []
        
#     # Método para filtrar objetos de interés dentro de la imagen
#     @staticmethod
#     @pxt.udf
#     def filter_target_objects(detections: list[list[float]]) -> list[list[float]]:
#         if not detections or not detections.get('classes'):
#             return {'boxes': [], 'classes': [], 'scores': [], 'target_found': False}
        
#         filtered_indices = []
#         for i, class_name in enumerate(detections['classes']):
#             if class_name.lower() in ['gun', 'rifle', 'pistol', 'weapon', 'knife', 'license plate', 'car', 'truck']:
#                 filtered_indices.append(i)

#         return {
#             'boxes': [detections['boxes'][i] for i in filtered_indices],
#             'classes': [detections['classes'][i] for i in filtered_indices],
#             'scores': [detections['scores'][i] for i in filtered_indices],
#             'target_found': len(filtered_indices) > 0
#         }

#     # Método que usa OCR para extraer texto de las patentes
#     @staticmethod
#     @pxt.udf
#     def extract_license_plate_text(img: PIL.Image.Image, boxes: list[list[float]]) -> list[str]:
#         if not boxes:
#             return []
#         texts = []
#         for box in boxes:
#             x1, y1, x2, y2 = [int(coord) for coord in box]
#             plate_img = img.crop((x1, y1, x2, y2))
#             plate_img = ImageOps.grayscale(plate_img)
#             plate_img = ImageEnhance.Contrast(plate_img).enhance(2.0)
#             try:
#                 text = pytesseract.image_to_string(plate_img, config='--psm 7 --oem 3').strip()
#                 texts.append(text if text else "No hay patente detectada")
#             except Exception as e:
#                 texts.append(f"OCR Error: {str(e)}")
#         return texts
    
#     # Configuración de las columnas computadas para las detecciones, 
#     # detecciones filtradas, patentes extraidas y embeddings para búsqueda
#     def setup_processing(self, extract_text=False, enable_search=False):
#         # Columnas para la detección
#         if 'raw_detections' not in self.table.columns:
#             self.table.add_computed_column(
#                 raw_detections=yolox(self.table.image, model_id=self.model, threshold=self.confidence_threshold)
#             )
        
#         if 'detections' not in self.table.columns:
#             self.table.add_computed_column(
#                 detections=self.filter_target_objects(self.table.raw_detections)
#             )
        
#         if extract_text and 'license_plate_text' not in self.table.columns:
#             self.table.add_computed_column(
#                 license_plate_text=self.extract_license_plate_text(self.table.image, self.table.detections.boxes)
#             )
        
#         # # Columna para la búsqueda
#         if enable_search:
#             self.table.add_embedding_index(
#                 'image',
#                 string_embed = sentence_transformer.using(model_id='intfloat/e5-large-v2'),
#                 image_embed= sentence_transformer.using(model_id='intfloat/e5-large-v2')
#             )

# -----------------------------------------------------------------------------------------------------------------------------

class VideoProcessor:
    def __init__(self, directory: str = "detection", 
                 yolox_model='yolox_m', 
                 confidence_threshold=0.25, 
                 fps=1,
                 embed_model: str ='openai/clip-vit-base-patch32'):
        self.directory = directory
        self.yolox_model = yolox_model
        self.confidence_threshold = confidence_threshold
        self.fps = fps
        self.embed_model = embed_model
        
    # Crea tabla de videos, view de frames, columnas de detecciones con yolox y embeddings para búsqueda
    # y extrae audio de los videos
    def setup(self) -> None:
        self.videos_table = pxt.create_table(self.directory, 
                                      {'video': pxt.VideoType()}, 
                                      if_exists='ignore')
        # View de frames
        self.frames_view = pxt.create_view(
            'frames',
            self.videos_table,
            iterator=FrameIterator.create(video=self.videos_table.video, fps=self.fps),
            if_exists='ignore'
        )
        
        self.videos_table.add_computed_column(audio_extract=extract_audio(self.videos_table.video, 
                                                                          format='mp3')) 
        
        self.frames_view.add_computed_column(
            raw_detections=yolox(self.frames_view.frame, model_id=self.yolox_model, threshold=self.confidence_threshold)
        )
        
        # Embedding para búsqueda (imágenes y texto)
        self.frames_view.add_embedding_index(
            'frame',
            string_embed=clip.using(model_id=self.embed_model, use_fast=True),
            image_embed=clip.using(model_id=self.embed_model, use_fast=True)
        )
        
        
    def process_videos(self, video_paths : list):
        video_objects = []
        for video_path in video_paths:
            video_objects.append({'video': video_path})
            
        # Más eficiete hacer un solo insert
        self.videos_table.insert(video_objects)

    def search_video(self, search_type : str = "Text", text_query=None, image_query=None, limit=5):
        try:
            if search_type == "Text" and text_query:
                sim = self.frames_view.frame.similarity(text_query)
            elif search_type == "Image" and image_query is not None:
                sim = self.frames_view.frame.similarity(image_query)
            else:
                return []
                
            results = self.frames_view.order_by(sim, asc=False).limit(limit).select(
                self.frames_view.frame, 
                self.frames_view.pos, 
                self.frframes_viewames.detections).collect()
            return list(results)
        except Exception as e:
            print(f"Error de búsqueda: {str(e)}")
            return []

    # # Configuración para video
    # def setup_processing(self, extract_text=False, enable_search=False):
    #     # Columnas para detección de objetos
    #     self.frames.add_computed_column(
    #         raw_detections=yolox(self.frames.frame, model_id=self.model, threshold=self.confidence_threshold)
    #     )
    #     self.frames.add_computed_column(
    #         detections=ImageProcessor.filter_target_objects(self.frames.raw_detections)
    #     )
    #     if extract_text:
    #         self.frames.add_computed_column(
    #             license_plate_text=ImageProcessor.extract_license_plate_text(self.frames.frame, self.frames.detections.boxes)
    #         )
    #     # Columna para la búsqueda
    #     if enable_search:
    #         self.frames.add_embedding_index('frame',
    #             string_embed=clip.using(model_id='openai/clip-vit-base-patch32', use_fast=True),
    #             image_embed=clip.using(model_id='openai/clip-vit-base-patch32', use_fast=True)
    #         )
    
    # # Método para buscar en frames de video
    # def search(self, search_type, text_query=None, image_query=None, limit=5):
    #     try:
    #         if search_type == "Text" and text_query:
    #             sim = self.frames.frame.similarity(text_query)
    #         elif search_type == "Image" and image_query is not None:
    #             sim = self.frames.frame.similarity(image_query)
    #         else:
    #             return []
                
    #         results = self.frames.order_by(sim, asc=False).limit(limit).select(
    #             self.frames.frame, 
    #             self.frames.pos, 
    #             self.frames.detections, 
    #             getattr(self.frames, 'license_plate_text', None)
    #         ).collect()
    #         return list(results)
    #     except Exception as e:
    #         print(f"Error de búsqueda: {str(e)}")
    #         return []
        
        
if __name__ == "__main__":
    print("HOLA")