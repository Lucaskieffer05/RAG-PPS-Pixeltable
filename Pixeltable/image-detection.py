import pixeltable as pxt
from pixeltable.ext.functions.yolox import yolox
import PIL.Image
import PIL.ImageDraw
import pytesseract # OCR
from PIL import ImageOps, ImageEnhance

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
DETECTION_TARGETS = ["gun", "license plate"]
EXTRACT_TEXT_FROM_LICENSE_PLATES = True

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

# Funciones definidas por el usuario 
@pxt.udf
def filter_target_objects(detections: list[list[float]]) -> list[list[float]]:
    """ Filtra las detecciones según la lista pasada"""
    
    # Si no encuentra nada devuelve un objeto vacío 
    if not detections or not detections.get('classes'):
        return {'boxes': [], 'classes': [], 'scores': [], 'target_found': False}
    
    # Si hay, las recorre en busca de patentes o armas
    filtered_indices = []
    for i, class_name in enumerate(detections['classes']):
        if (('gun' in DETECTION_TARGETS and class_name.lower() in ['gun', 'rifle', 'pistol', 'weapon', 'knife']) or
            ('license plate' in DETECTION_TARGETS and class_name.lower() in ['license plate', 'car', 'truck'])):
            filtered_indices.append(i)
    
    # Arma el objeto de resultado
    result = {
        'boxes': [detections['boxes'][i] for i in filtered_indices],
        'classes': [detections['classes'][i] for i in filtered_indices],
        'scores': [detections['scores'][i] for i in filtered_indices],
        'target_found': len(filtered_indices) > 0
    }
    
    return result




@pxt.udf
def extract_license_plate_text(img: PIL.Image.Image, boxes: list[list[float]]) -> list[str]:
    """Extrae texto de las patentes identificadas"""
    if not boxes:
        return []
    
    # Para cada una de las detecciones arma el recuadro que contiene la patente e intenta obtener su info
    texts = []
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        plate_img = img.crop((x1, y1, x2, y2))
        
        # Pre-procesamiento        
        plate_img = ImageOps.grayscale(plate_img)
        plate_img = ImageEnhance.Contrast(plate_img).enhance(2.0)
        
        # TODO: Ver configuración de parámetros
        try:
            text = pytesseract.image_to_string(plate_img, config='--psm 7 --oem 3').strip()
            texts.append(text if text else "No hay patente detectada")
        except Exception as e:
            texts.append(f"OCR Error: {str(e)}")
    
    return texts

# TODO: Revisar función
@pxt.udf
def draw_boxes_with_labels(img: PIL.Image.Image, detections) -> PIL.Image.Image:
    """Dibuja boxes con sus correspondientes labels"""
    if not detections or not detections.get('boxes'):
        return img
        
    result = img.copy()
    d = PIL.ImageDraw.Draw(result)
    
    for i, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = box
        
        d.rectangle(box, outline="red", width=3)
        
        label = f"{detections['classes'][i]}: {detections['scores'][i]:.2f}"
        d.text((x1, y1-15), label, fill="red")
        
        if 'license_plate_text' in globals() and i < len(license_plate_text):
            d.text((x1, y2+5), license_plate_text[i], fill="blue")
            
    return result

# Agrega la visualización a ambas tablas (imágenes y videos)
images.add_computed_column(
    visualization=draw_boxes_with_labels(images.image, images.detections.boxes)
)

frames.add_computed_column(
    visualization=draw_boxes_with_labels(frames.frame, frames.detections.boxes)
)

# Columnas computadas para la detección (se pasa el modelo y el umbral de confianza)

# IMÁGENES ----------------
images.add_computed_column(
    detections=yolox(
        images.image,
        model_id=SELECTED_MODEL_IMAGES,  
        threshold=CONFIDENCE_TRESHOLD_IMAGES
    )
)

# Columna para las detecciones
images.add_computed_column(
    detections=filter_target_objects(images.raw_detections)
)

# Columna para la data de patentes
if EXTRACT_TEXT_FROM_LICENSE_PLATES:
    images.add_computed_column(
        license_plate_text=extract_license_plate_text(images.image, images.detections.boxes)
    )

# FRAMES ----------------

frames.add_computed_column(
    detections=yolox(
        frames.frame,
        model_id=SELECTED_MODEL_VIDEOS,
        threshold=CONFIDENCE_TRESHOLD_VIDEOS
    )
)

frames.add_computed_column(
    detections=filter_target_objects(frames.raw_detections)
)

if EXTRACT_TEXT_FROM_LICENSE_PLATES:
    frames.add_computed_column(
        license_plate_text=extract_license_plate_text(frames.frame, frames.detections.boxes)
    )


