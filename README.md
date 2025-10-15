# Porto Seguro’s Safe Driver Prediction – Proyecto Académico

##  Resumen
Este proyecto académico replica la competencia de Kaggle **Porto Seguro’s Safe Driver Prediction**, cuyo objetivo es **nuestra participacion y predecir si un conductor realizará un reclamo de seguro en el próximo año**.  
El enfoque principal es **modelado de datos tabulares, manejo de features y evaluación mediante métricas de ranking**, específicamente la **Normalized Gini Coefficient**.

**Modelo final:** Logistic Regression  
**Número de features usadas:** 95  
**Predicciones generadas:** 50,000  
**Archivo listo para enviar:** `submission.csv`

---

##  Datos
- Datos originales de Kaggle **no incluidos** por restricciones de licencia.
- Conjunto original:
  - Train: 595,212 filas × 59 columnas
  - Test: 892,816 filas × 58 columnas
- **Muestreo estratificado** para trabajar con datasets más manejables:
  - Train reducido: 100,000 filas (manteniendo la proporción del target)
  - Test reducido: 50,000 filas
  - Proporción de reclamos (target = 1) en la muestra: 3.65%

```python
# Ejemplo de muestreo estratificado
from sklearn.model_selection import train_test_split

train_sample, _ = train_test_split(
    train, 
    train_size=100000, 
    stratify=train['target'], 
    random_state=42
)

test_sample = test.sample(n=50000, random_state=42)

train = train_sample.copy()
test = test_sample.copy()
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Proporción target: {train['target'].mean():.2%}")

```
Nota: El dataset original es muy grande, lo que puede hacer que el notebook se trabe. Este muestreo permite entrenar y experimentar más rápido sin perder la proporción de la clase objetivo.

---
##  Proceso de trabajo

### 1️ Exploración de datos (EDA)
- Análisis de distribución de variables y correlaciones.
- Identificación de valores faltantes y outliers.

### 2️ Preprocesamiento
- Imputación de valores faltantes.
- Normalización y codificación de variables categóricas.

### 3️ Selección de features
- Basada en análisis estadístico y correlación.
- **95 features finales seleccionadas**.

### 4️ Modelado
- Modelo final: **Logistic Regression**
- Validación cruzada: **Stratified K-Fold**
- Evaluación basada en **Normalized Gini** y **ROC-AUC**

### 5️ Resultados

| Métrica           | Cross Validation | Validación |
|------------------|-----------------|------------|
| Normalized Gini   | 0.2420          | 0.2476     |
| ROC-AUC           | 0.6210          | 0.6238     |

> El modelo muestra **capacidad moderada de ranking de riesgo**, suficiente para fines educativos y aprendizaje de técnicas de machine learning.

##  Aprendizajes
- Manejo de **datos desbalanceados** (solo 3.65% de positivos).  
- Importancia de las **métricas de ranking** frente a métricas clásicas de clasificación.  
- Implementación de **pipeline completo de ML**: exploración --> preprocesamiento --> modelado --> predicciones.  
- Documentar claramente **procesos y resultados** para garantizar reproducibilidad.  

##  Cómo reproducir
1. Clonar el repositorio.  
2. Preparar un **dataset de ejemplo** siguiendo la estructura de columnas.  
3. Ejecutar `notebook.ipynb` para reproducir el preprocesamiento, entrenamiento y generación de predicciones.

