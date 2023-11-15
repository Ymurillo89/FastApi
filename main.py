from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from PIL import Image
from io import BytesIO
import re
import joblib
import cv2
import pandas as pd
import numpy as np
from skimage import io, color, feature
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Directorio donde se encuentra el modelo KNN entrenado
model_path = 'assets/modelFruits.pkl'
scaler_path = 'assets/escaladorFruits.pkl'
labels_path = 'assets/labelFruits.joblib'

# Cargar el modelo, escalador y etiquetas
knn_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
labels = joblib.load(labels_path)

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost",  # Reemplaza con la URL de tu aplicación Angular
    "http://localhost:4200",  # Ejemplo para Angular en desarrollo
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image: str

@app.post("/uploadfile")
async def upload_file(image_request: ImageRequest):
   
    # Obtener el tipo de imagen (por ejemplo, 'image/jpeg')
    image_type = re.search(r'^data:image/(\w+);base64,', image_request.image).group(1)

    # Obtener los datos base64 (sin el encabezado)
    image_data_base64 = re.sub(r'^data:image/\w+;base64,', '', image_request.image)

    try:
        # Decodificar la cadena base64 a una representación binaria
        image_data = base64.b64decode(image_data_base64)

        # Crear una imagen a partir de los datos binarios
        image = Image.open(BytesIO(image_data))       

        # Convertir de RGB a BGR
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convertir la imagen BGR a escala de grises
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Calcular el Local Binary Pattern (LBP)
        lbp = feature.local_binary_pattern(image_gray, P=8, R=1, method='uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=[0, 9])

        image_bgr = np.array(image)[:, :, :3]  # Considerar solo los canales RGB
        hist_bgr = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_bgr = hist_bgr.flatten()
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_hsv = hist_hsv.flatten()
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        hist_lab = cv2.calcHist([image_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_lab = hist_lab.flatten()
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(image_gray, P=8, R=1, method='uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=[0, 9])
        size_feature = image.size
        features = list(hist_bgr) + list(hist_hsv) + list(hist_lab) + list(hist_lbp) + [size_feature]

        #print(len(features))
        # Normalizar características usando el escalador        
        features_normalized = scaler.fit_transform(np.array(features[:-1]).reshape(1, -1))
     
        # Aplicar KNN para clasificar la nueva señal de audio
        knn = KNeighborsClassifier(n_neighbors=5)  # Utiliza el mismo número de vecinos que se usó en el modelo entrenado
        # Carga las etiquetas correspondientes (y_train)
        X_train = joblib.load('assets/modelFruits.pkl')
        y_train_dict = joblib.load('assets/labelFruits.joblib')

        # Transformar el diccionario de etiquetas a una lista
        y_train = [y_train_dict[fruit] for fruit in y_train_dict.keys()]

        print(X_train)
        print(y_train)

        knn.fit(X_train, y_train_dict)
       
        # Responder al cliente con un mensaje o los resultados del procesamiento
        return {"message": "Imagen procesada exitosamente"}
    except PIL.UnidentifiedImageError as e:
        return {"error": "No se pudo identificar la imagen. Verifica el formato y los datos de la imagen."}

# Resto del código...
