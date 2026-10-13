# Trabajo Práctico 1 - Ciencia de Datos (TA047)
**Facultad de Ingeniería, Universidad de Buenos Aires (FIUBA)**  
**Cátedra:** Dr. Ing. Rodríguez  
**Grupo:** 14  
**Ciudad Asignada (Ej. 2):** Toronto  
---
## Integrantes del Grupo
* **Calandria, Alén Santiago 112103**
* **Szinvelski, Sol Yohara 112985**
* **Garay Alvarado, Gustavo Alfredo .intercambio-02**
* **Sosa, Matias 113352**
* **Fontanals Layos, Oriol .intercambio-03**
---
## Contenido del Repositorio
El repositorio está organizado conforme a las pautas de entrega del Trabajo Práctico:
* **`notebooks/`**: Notebooks de Jupyter con el desarrollo completo, explicaciones y resultados ejecutados de forma reproducible.
  * `TA047R_TP1_GRUPO14_ENTREGA_EJ1.ipynb`: Análisis exploratorio sobre Rebrickable (LEGO) y clustering no supervisado mediante K-Means.
  * `TA047R_TP1_GRUPO14_ENTREGA_EJ2.ipynb`: Preprocesamiento, ingeniería de features y modelos de regresión (Lineal y Random Forest con optimización por Cross Validation) para predecir precios de alojamientos en Toronto (Inside Airbnb).
  * `TA047R_TP1_GRUPO14_ENTREGA_EJ3.ipynb`: Clasificación supervisada de reservas hoteleras para la competencia de Kaggle (Árbol de Decisión, Random Forest, XGBoost, Voting y Stacking), análisis de importancia de variables, matrices de confusión e interpretación de árboles.
* **`reporte/`**:
  * `TA047R_TP1_GRUPO14_REPORTE.pdf`: Informe final del trabajo práctico siguiendo la estructura y pautas de la cátedra.
* **`modelos/`**: Modelos entrenados y serializados (`.joblib`) para cada uno de los ejercicios.
* **`datos/`**:
  * `rebrickable/`: Datasets originales utilizados en el Ejercicio 1.
  * `airbnb/`: Dataset de alojamientos de Toronto.
  * `hoteles/`: Datasets oficiales provistos para la competencia de Kaggle (`train`, `test` y `sample_submission`).
  * `procesados/`: Datasets analíticos y particiones intermedias generadas durante el preprocesamiento.
* **`src/`**: Módulo Python auxiliar (`hotels_features.py`) requerido para la deserialización y ejecución del pipeline de características en los modelos del Ejercicio 3.

---
## Requisitos e Instalación
Para ejecutar y reproducir los análisis en un entorno local, se recomienda utilizar Python 3.12:
1. **Crear y activar un entorno virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(En Linux / macOS: `source .venv/bin/activate`)*
2. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Iniciar el entorno de trabajo:**
   ```bash
   jupyter lab
   ```
> **Nota:** Todos los notebooks se encuentran guardados con sus salidas y celdas ejecutadas para facilitar su lectura y corrección directa.
