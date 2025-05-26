import pixeltable as pxt
from pixeltable.iterators import FrameIterator
from pixeltable.functions.video import extract_audio

from pixeltable.functions.huggingface import clip
from pixeltable.ext.functions.yolox import yolox
import PIL.Image


# -----------------------------------------------------------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self, directory: str = "image_processor_data", 
                 yolox_model: str = 'yolox_m', 
                 confidence_threshold: float = 0.25, 
                 embed_model: str = 'openai/clip-vit-base-patch32'):
        self.directory = directory
        self.yolox_model = yolox_model
        self.confidence_threshold = confidence_threshold
        self.embed_model = embed_model
        self.images_table = None
        
    def setup(self) -> None:
        self.images_table = pxt.create_table(
            self.directory, 
            {'image': pxt.Image}, 
            if_exists='replace_force'
        )
        
        if 'raw_detections' not in self.images_table.columns:
            self.images_table.add_computed_column(
                raw_detections=yolox(self.images_table.image, model_id=self.yolox_model, threshold=self.confidence_threshold)
            )
        
        # Embedding para búsqueda (imágenes y texto)
        self.images_table.add_embedding_index(
            'image', 
            string_embed=clip.using(model_id=self.embed_model), 
            image_embed=clip.using(model_id=self.embed_model)  
        )
        
        if hasattr(self.images_table, 'get_info'):
            info = self.images_table.get_info()
            print("Available indices on images_table:", info.get('indices', 'Info method not providing index details'))
        
    def process_images(self, image_paths : list):
        image_objects = []
        for image_path in image_paths:
            try:
                with open(image_path, 'rb') as f:
                    pass
                image_objects.append({'image': image_path})
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_path}, skipping.")
            except Exception as e:
                print(f"Warning: Could not process image path {image_path}: {e}, skipping.")

        if not image_objects:
            print("No valid image paths provided to process.")
            return
            
        self.images_table.insert(image_objects)

    def search_image(self, search_type : str = "Text", text_search_query=None, image_query=None, limit=5):
        if self.images_table is None:
            print("Error: Images table not set up. Please run setup() first.")
            return []

        if hasattr(self.images_table, 'get_info'):
            info = self.images_table.get_info()
            print("Available search indices on images_table:", info.get('indices', 'No index info available'))

        try:
            query_input = None
            if search_type == "Text" and text_search_query:
                query_input = text_search_query
            elif search_type == "Image" and image_query is not None:
                if isinstance(image_query, str):
                    try:
                        query_input = PIL.Image.open(image_query)
                    except FileNotFoundError:
                        print(f"Error: Query image file not found at {image_query}")
                        return []
                    except Exception as e:
                        print(f"Error: Could not open query image {image_query}: {e}")
                        return []
                elif isinstance(image_query, PIL.Image.Image):
                    query_input = image_query
                else:
                    print("Error: Invalid image_query type. Must be a file path or PIL.Image object.")
                    return []
            else:
                print("Error: Search type not supported or query not provided.")
                return []
            
            if query_input is None: 
                print("Error: Query input is None.")
                return []

            sim = self.images_table.image.similarity(query_input)
                
            results = self.images_table.order_by(sim, asc=False).limit(limit).select(
                self.images_table.image, 
                self.images_table.raw_detections
            ).collect()
            return list(results)
        except Exception as e:
            print(f"Error de búsqueda de imagen: {str(e)}")
            return []
        
# -----------------------------------------------------------------------------------------------------------------------------

class VideoProcessor:
    def __init__(self, directory: str = "detection", 
                 yolox_model: str = 'yolox_m', 
                 confidence_threshold: float = 0.25, 
                 fps: int = 10,
                 embed_model: str = 'openai/clip-vit-base-patch32'):
        self.directory = directory
        self.yolox_model = yolox_model
        self.confidence_threshold = confidence_threshold
        self.fps = fps
        self.embed_model = embed_model
        self.videos_table = None
        self.frames_view = None
        
    # Crea tabla de videos, view de frames, columnas de detecciones con yolox y embeddings para búsqueda
    # y extrae audio de los videos
    def setup(self) -> None:
        self.videos_table = pxt.create_table(self.directory, 
                                      {'video': pxt.VideoType()}, # Esto debería ser pxt.Video
                                      if_exists='replace_force')
        # View de frames
        self.frames_view = pxt.create_view(
            'frames',
            self.videos_table,
            iterator=FrameIterator.create(video=self.videos_table.video, fps=self.fps),
            if_exists='replace_force'
        )
        
        if 'audio_extract' not in self.videos_table.columns:
            self.videos_table.add_computed_column(audio_extract=extract_audio(self.videos_table.video, 
                                                                          format='mp3')) 
        if 'raw_detections' not in self.frames_view.columns:
            self.frames_view.add_computed_column(
                raw_detections=yolox(self.frames_view.frame, model_id=self.yolox_model, threshold=self.confidence_threshold)
            )
        
        # Embedding para búsqueda (imágenes y texto)
        self.frames_view.add_embedding_index(
            'frame',
            string_embed=clip.using(model_id=self.embed_model),
            image_embed=clip.using(model_id=self.embed_model)
        )
        
        if hasattr(self.frames_view, 'get_info'):
            info = self.frames_view.get_info()
            print("Available indices:", info.get('indices', 'Info method not providing index details'))
        
    def process_videos(self, video_paths : list):
        video_objects = []
        for video_path in video_paths:
            video_objects.append({'video': video_path})
            
        # Más eficiete hacer un solo insert
        self.videos_table.insert(video_objects)

    def search_video(self, search_type : str = "Text", text_search_query=None, image_query=None, limit=5):
        
        if hasattr(self.frames_view, 'get_info'):
            info = self.frames_view.get_info()
            print("Available search indices:", info.get('indices', 'No index info available'))

        try:
            if search_type == "Text" and text_search_query:
                sim = self.frames_view.frame.similarity(text_search_query)
            elif search_type == "Image" and image_query is not None:
                sim = self.frames_view.frame.similarity(image_query)
            else:
                return []
                
            results = self.frames_view.order_by(sim, asc=False).limit(limit).select(
                self.frames_view.frame, 
                self.frames_view.pos, 
                self.frames_view.raw_detections
            ).collect()
            return list(results)
        except Exception as e:
            print(f"Error de búsqueda: {str(e)}")
            return []
        
        
if __name__ == "__main__":
    print("HOLA")