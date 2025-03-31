import pixeltable as pxt
from pixeltable.ext.functions.yolox import yolox
import PIL.Image
import PIL.ImageDraw

# Lista de modelos (ordenados por calidad - de menor a mayor)
  # yolox_nano
  # yolox_tiny
  # yolox_s
  # yolox_m
  # yolox_l
  # yolox_x

# PARÁMETROS
SELECTED_MODEL_IMAGES = 'yolox_s'
CONFIDENCE_TRESHOLD_IMAGES = 0.5
CONFIDENCE_TRESHOLD_VIDEOS = 0.25
SELECTED_MODEL_VIDEOS = 'yolox_m'
EXTRACTED_FRAMES_PER_SECOND = 1 # Cuantos frames del video se toman por segundo

# Iniciar entorno
pxt.drop_dir('detection')
pxt.create_dir('detection')

# Definir tablas para los archivos
images = pxt.create_table(
  'detection.images',
  {'image': pxt.ImageType()},
  if_exists='ignore'
)

videos = pxt.create_table(
  'detection.videos',
  {'video': pxt.VideoType()},
  if_exists='ignore'
)

# La view de frames se configura para trackear automáticamente los videos a partir de que llega
# nueva data. Se configura con seis columnas
  # pos - una columna que es parte de cada view
  # video - la columna que hereda de la tabla 
  # frame_idx, pos_msec, pos_frame, frame - se generan automáticamente por la clas frameIterator
frames = pxt.create_view(
    'detection.frames',
    videos,
    iterator=pxt.iterators.FrameIterator.create(
        video=videos.video,
        fps=EXTRACTED_FRAMES_PER_SECOND
    )
)

# Columnas computadas para la detección (se pasa el modelo y el umbral de confianza)

images.add_computed_column(
    detections=yolox(
        images.image,
        model_id=SELECTED_MODEL_IMAGES,  
        threshold=CONFIDENCE_TRESHOLD_IMAGES
    )
)

frames.add_computed_column(
    detections=yolox(
        frames.frame,
        model_id=SELECTED_MODEL_VIDEOS,
        threshold=CONFIDENCE_TRESHOLD_VIDEOS
    )
)

# Función para visualizar las detecciones en las imágenes
@pxt.udf
def draw_boxes(img: PIL.Image.Image, boxes: list[list[float]]) -> PIL.Image.Image:
    result = img.copy()
    d = PIL.ImageDraw.Draw(result)
    for box in boxes:
        d.rectangle(box, width=3)
    return result

# Agrega la visualización a ambas tablas (imágenes y videos)
images.add_computed_column(
    visualization=draw_boxes(images.image, images.detections.boxes)
)

frames.add_computed_column(
    visualization=draw_boxes(frames.frame, frames.detections.boxes)
)