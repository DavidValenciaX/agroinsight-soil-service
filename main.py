from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import asyncio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image
import io
import timm

# En la configuración de FastAPI
app = FastAPI(
    title="Análisis de Suelos API",
    description="API para analizar suelos",
    version="1.0.0"
)

# Agregar límite de tamaño a nivel de FastAPI
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Ajustar según necesidades
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    # Limitar el tamaño total de la petición
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_IMAGE_SIZE * MAX_IMAGES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Petición demasiado grande"}
            )
    response = await call_next(request)
    return response

# Lista de clases de suelos
CLASSES = ["Alluvial Soil", "Black Soil", "Cinder Soil", "Clay Soil", "Laterite Soil", "Peat Soil", "Yellow Soil"]

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
])

def preprocess_image(image):
    # Aplicar transformaciones
    img_tensor = transform(image)
    # Agregar dimensión de batch
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# Cargar el modelo al iniciar la aplicación
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recrear el modelo
model_name = 'caformer_s18.sail_in22k_ft_in1k_384'
model = timm.create_model(model_name, pretrained=False)

# Configurar el modelo para que coincida con el número de clases
num_classes = len(CLASSES)
in_features = model.head.fc.fc1.in_features

# Crear una nueva cabeza más simple manteniendo las capas necesarias
class NewHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.global_pool = model.head.global_pool
        self.norm = model.head.norm
        self.flatten = model.head.flatten
        self.drop = model.head.drop
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x

model.head = NewHead(in_features, num_classes)

# Cargar el modelo al iniciar la aplicación
def load_model():
    try:
        # Verificar que el archivo existe
        import os
        model_path = 'pesos_modelo_identificacion_suelos.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encuentra el archivo de pesos: {model_path}")
            
        # Verificar el tamaño del archivo
        file_size = os.path.getsize(model_path)
        print(f"Tamaño del archivo de pesos: {file_size} bytes")
        
        # Intentar cargar el modelo
        state_dict = torch.load(model_path, map_location=device)
        
        # Verificar que el state_dict tiene las claves esperadas
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        if expected_keys != loaded_keys:
            print("Advertencia: Las claves del modelo no coinciden")
            print("Claves faltantes:", expected_keys - loaded_keys)
            print("Claves extras:", loaded_keys - expected_keys)
            
        model.load_state_dict(state_dict)
        print("Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise

# Cargar el modelo
model = load_model()
model.to(device)
model.eval()

async def process_single_image(image_data, filename: str):
    try:
        # Abrir la imagen en memoria
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocesar la imagen
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # Realizar la predicción
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            probabilities = probabilities[0].tolist()
        
        # Limpiar memoria explícitamente
        del image
        del img_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "filename": filename,
            "status": "success",
            "predicted_class": CLASSES[predicted_class],
            "confidence": probabilities[predicted_class],
            "probabilities": {
                class_name: prob 
                for class_name, prob in zip(CLASSES, probabilities)
            }
        }
        
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "error": str(e)
        }

# Agregar constantes de límites
MAX_IMAGES = 15  # Máximo número de imágenes por petición
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB por imagen
SUPPORTED_FORMATS = {'image/jpeg', 'image/png'}

@app.post("/predict")
async def predict_multiple(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se han proporcionado archivos"
        )
    
    # Validar número de imágenes
    if len(files) > MAX_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Número máximo de imágenes permitido es {MAX_IMAGES}"
        )
    
    try:
        results = []
        tasks = []
        
        for file in files:
            # Validar formato
            if file.content_type not in SUPPORTED_FORMATS:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "Formato no soportado. Use JPG o PNG"
                })
                continue
            
            # Validar tamaño antes de leer el archivo
            image_data = await file.read()
            if len(image_data) > MAX_IMAGE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"Imagen demasiado grande. Máximo {MAX_IMAGE_SIZE/1024/1024}MB"
                })
                continue
                
            task = asyncio.create_task(process_single_image(image_data, file.filename))
            tasks.append(task)
        
        # Procesar en lotes si hay muchas imágenes
        BATCH_SIZE = 3  # Procesar máximo 3 imágenes simultáneamente
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i:i + BATCH_SIZE]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
            # Pequeña pausa entre lotes para evitar sobrecarga
            if i + BATCH_SIZE < len(tasks):
                await asyncio.sleep(0.1)
        
        # Analizar resultados para determinar el código de estado
        successful_predictions = [r for r in results if r["status"] == "success"]
        failed_predictions = [r for r in results if r["status"] == "error"]
        
        if not results:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "message": "No se pudo procesar ninguna imagen",
                    "results": []
                }
            )
        
        if len(successful_predictions) == len(results):
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "Todas las imágenes fueron procesadas exitosamente",
                    "results": results
                }
            )
        
        if len(failed_predictions) == len(results):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "message": "No se pudo procesar ninguna imagen",
                    "results": results
                }
            )
        
        return JSONResponse(
            status_code=status.HTTP_207_MULTI_STATUS,
            content={
                "message": "Algunas imágenes no pudieron ser procesadas",
                "successful": len(successful_predictions),
                "failed": len(failed_predictions),
                "results": results
            }
        )
            
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Error interno del servidor",
                "error": str(e)
            }
        )

@app.get("/")
async def root():
    return {
        "message": "API de Análisis de Suelos",
        "endpoints": {
            "/predict": "POST - Envía un lote de imágenes para analizar",
            "/": "GET - Muestra esta información"
        }
    } 
    
@app.get("/health")
async def health_check():
    """Endpoint para verificar que el servicio está funcionando"""
    try:
        # Intentar cargar el modelo para verificar que está disponible
        model.eval()
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(device)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }