
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from PIL import Image
from io import BytesIO
import re
import joblib



# Directorio donde se encuentra el modelo KNN entrenado
model_path = 'assets/Modelo_faces_KNN.pkl'

#Cargar el modelo
knn_model = joblib.load(model_path)

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

        # Responder al cliente con un mensaje o los resultados del procesamiento
        return {"message": "Imagen procesada exitosamente"}
    except PIL.UnidentifiedImageError as e:
        return {"error": "No se pudo identificar la imagen. Verifica el formato y los datos de la imagen."}


@app.get("/hello")
def hello_world():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)