import pixeltable as pxt

# Conectar a las tablas
images = pxt.get_table("detection.images")
videos = pxt.get_table("detection.videos")
frames = pxt.get_table("detection.frames")

# Procesar imágenes
images.insert([
    {'image': 'path/to/image1.jpg'},
    {'image': 'path/to/image2.jpg'}
])

# Procesar videos
videos.insert([
    {'video': 'path/to/video1.mp4'}
])

# Realizar detección
image_results = images.select(
    images.image,
    images.detections,
    images.visualization
).collect()

frame_results = frames.select(
    frames.frame,
    frames.detections,
    frames.visualization
).collect()

# Acceder a información específica de la detección
high_confidence = frames.where(
    frames.detections.scores[0] > 0.9
).collect()