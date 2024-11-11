FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar primero solo el archivo de pesos y verificar su existencia
COPY pesos_modelo_identificacion_suelos.pth .
RUN ls -l pesos_modelo_identificacion_suelos.pth

# Copiar el resto del código
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"] 