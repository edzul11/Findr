# Findr - Sistema de Rastreo y Búsqueda Semántica de Personas

Este proyecto combina la potencia de **YOLO** para la detección y rastreo de personas en tiempo real, junto con **OpenAI CLIP** y **FAISS** para permitir la búsqueda semántica de las personas detectadas utilizando descripciones en lenguaje natural.

## 🚀 Funcionalidades

*   **Detección y Rastreo en Tiempo Real**: Utiliza un modelo YOLO (v12) para detectar y rastrear personas desde una fuente de video.
*   **Búsqueda Semántica**: Permite buscar personas específicas dentro de las detecciones almacenadas utilizando descripciones de texto (ej. "persona con camisa roja y sombrero").
*   **Base de Datos Vectorial**: Emplea FAISS para indexar y buscar eficientemente entre los embeddings de las imágenes generados por CLIP.
*   **Interfaz Web**: Incluye una interfaz gráfica construida con **Streamlit** para gestionar la base de datos y realizar búsquedas de manera intuitiva.

## 🛠️ Tecnologías Utilizadas

*   **Python 3.x**
*   **Ultralytics YOLO**: Para detección de objetos (v12).
*   **Sentence-Transformers (CLIP)**: Modelo `clip-ViT-B-16` para generar embeddings de imágenes y texto.
*   **FAISS**: Librería de Facebook para búsqueda de similitud eficiente.
*   **Streamlit**: Para la interfaz de usuario web.
*   **Torch & Torchvision**: Framework de Deep Learning.

## 📦 Instalación

1.  Clona este repositorio:
    ```bash
    git clone https://github.com/tu-usuario/nombre-repo.git
    cd nombre-repo
    ```

2.  Crea y activa un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    # En Windows:
    venv\Scripts\activate
    # En Linux/Mac:
    source venv/bin/activate
    ```

3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Uso

### 1. Detección y Rastreo
Ejecuta el script principal para iniciar el rastreo de personas:
```bash
python main.py
```
Este script iniciará la cámara (índice 1 por defecto) y comenzará a detectar personas, guardando las capturas en la carpeta `imagenes/personas_detectadas`.

### 2. Búsqueda de Personas
Inicia la interfaz web para buscar en la base de datos de personas detectadas:
```bash
streamlit run buscar32.py
```
Desde la interfaz podrás:
*   **Actualizar Base de Datos**: Procesar nuevas imágenes detectadas para añadirlas al índice.
*   **Buscar**: Ingresa una descripción (ej. "hombre con gafas") para encontrar coincidencias.
