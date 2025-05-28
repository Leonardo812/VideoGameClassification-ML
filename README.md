# VideoGameClassification-ML

## 🎮 Clasificación de Videojuegos mediante Machine Learning

Este proyecto tiene como objetivo desarrollar un modelo capaz de identificar distintos videojuegos a partir de imágenes, utilizando técnicas de Aprendizaje Automático (Machine Learning). Esta clasificación puede ser útil para sistemas de recomendación, organización de bibliotecas de juegos o análisis de contenido multimedia relacionado con videojuegos.

Se emplea **Aprendizaje Supervisado**, una rama del Machine Learning en la que el modelo aprende a reconocer patrones a partir de un conjunto de datos etiquetado. A través del entrenamiento con estas imágenes, el modelo es capaz de predecir correctamente la categoría o el videojuego correspondiente de nuevas imágenes no vistas.

---

## 🕹️ Clases del Modelo

El modelo ha sido entrenado para reconocer y clasificar imágenes de distintos videojuegos populares, cada uno con características visuales y estéticas particulares. Estas son las categorías de videojuegos que el modelo identifica, junto con una breve descripción de cada una:

- **Dead by Daylight**  
  Juego multijugador asimétrico (1 vs 4) de horror y supervivencia. Un jugador asume el rol de asesino, mientras los demás intentan sobrevivir.  
  **Estilo visual:** Entornos oscuros, iluminación tenue y predominancia de colores opacos y sombríos.  
  **Perspectiva:** 3D en tercera persona.

- **Counter-Strike: Global Offensive (CS:GO)**  
  Shooter táctico multijugador por equipos (5v5), centrado en enfrentamientos entre terroristas y antiterroristas.  
  **Estilo visual:** Ambientes urbanos o industriales, paleta de colores realista y sobria.  
  **Perspectiva:** 3D en primera persona.

- **Valorant**  
  Shooter táctico multijugador que combina mecánicas de disparo con habilidades únicas por personaje.  
  **Estilo visual:** Gráficos estilizados con colores vivos, mapas bien iluminados y efectos visuales llamativos.  
  **Perspectiva:** 3D en primera persona.

- **Clash Royale**  
  Juego de estrategia en tiempo real para móviles, basado en cartas coleccionables y batallas rápidas entre dos jugadores.  
  **Estilo visual:** Estética caricaturesca en 2.5D, colores brillantes y animaciones simplificadas.  
  **Perspectiva:** Vista cenital (de arriba hacia abajo).

- **Minecraft**  
  Juego de construcción y supervivencia en un mundo abierto generado por bloques.  
  **Estilo visual:** Estética pixelada en 3D, colores variados pero con texturas simples y cuadradas.  
  **Perspectiva:** Puede alternar entre primera y tercera persona.

- **Overwatch**  
  Shooter por equipos con enfoque en héroes, cada uno con habilidades y roles específicos.  
  **Estilo visual:** Gráficos coloridos y estilizados, ambientaciones futuristas y personajes visualmente distintivos.  
  **Perspectiva:** 3D en primera persona.

---

## 📁 Generación y selección del conjunto de datos

Para este proyecto se utilizó el dataset [**Videogame Image Classification**](https://www.kaggle.com/datasets/juanmartinzabala/videogame-image-classification?resource=download-directory) disponible en Kaggle, el cual contiene capturas de pantalla de diversos videojuegos populares.

### 🔍 Proceso de limpieza y selección

A partir del conjunto original, se realizó una limpieza significativa, seleccionando únicamente 6 de las 21 clases disponibles. Esta decisión se tomó porque muchas clases presentaban imágenes con problemas como:

- Capturas limitadas a una sola zona del videojuego.
- Presencia de elementos no relacionados (como cámaras de streamers o superposiciones gráficas de transmisiones).
- Poca diversidad en escenarios y contenido visual del juego.

Estas limitaciones podían afectar negativamente la capacidad del modelo para generalizar, por lo que se optó por trabajar solo con clases visualmente más consistentes y representativas.

### 📦 Reducción y balanceo del dataset

Cada clase del dataset original contenía aproximadamente 4,000 imágenes para entrenamiento y 1,000 para prueba. No obstante, se redujo intencionalmente la cantidad de imágenes a un aproximado de:

- **700 imágenes por clase para entrenamiento**
- **150 imágenes por clase para prueba**
  
Adicionalmente se uso el 10% de datos de cada clase como datos de validación.

Esta reducción tuvo como objetivo priorizar la calidad y la diversidad del contenido, manteniendo un balance entre clases y una proporción de **70% para entrenamiento, 10% para validación y 20% para prueba**.

---

## 📊 Distribución de los datos

Después del proceso de limpieza y selección, el conjunto de datos quedó conformado por un total de **4,451 imágenes para entrenamiento**, distribuidas de manera balanceada entre las seis clases seleccionadas:

| Videojuego           | Imágenes (train) |
|----------------------|------------------|
| Dead by Daylight     | 704              |
| Counter-Strike       | 733              |
| Valorant             | 789              |
| Clash Royale         | 718              |
| Minecraft            | 787              |
| Overwatch            | 720              |
| **Total**            | **4,451**        |

| Videojuego           | Imágenes (test) |
|----------------------|------------------|
| Dead by Daylight     | 157              |
| Counter-Strike       | 154              |
| Valorant             | 165              |
| Clash Royale         | 161              |
| Minecraft            | 157              |
| Overwatch            | 151              |
| **Total**            | **945**        |

Esta distribución permite mantener una representación relativamente equilibrada entre las clases, lo que ayuda a que el modelo aprenda patrones visuales distintivos de cada videojuego. En caso de ser necesario, se podrá incrementar el número de imágenes por clase para mejorar el rendimiento del modelo o reforzar aquellas categorías que presenten menor precisión durante las pruebas.

## 🛠️ Preprocesado y técnicas de aumento de datos

Para mejorar la capacidad del modelo para generalizar y hacerlo más robusto frente a las distintas condiciones gráficas en las que pueden presentarse los videojuegos (por ejemplo, baja calidad, capturas con diferente iluminación o resolución), se aplicaron las siguientes técnicas de preprocesamiento:

### 🔹 Reescalado (normalización)
Cada valor de píxel se divide entre **255** (`rescale=1./255`) para transformar el rango de los píxeles de **[0, 255] a [0, 1]**. Esto estabiliza el proceso de optimización y mejora la convergencia del modelo durante el entrenamiento.

### 🔹 Redimensionamiento (resize)
Todas las imágenes se redimensionaron al tamaño de **224x224 píxeles**, que es un estándar común para modelos de redes neuronales convolucionales (CNN). Esto asegura consistencia en la entrada de datos.

### 🔹 Aumento de datos (data augmentation)
Se aplicaron transformaciones aleatorias a las imágenes del conjunto de entrenamiento para simular una mayor diversidad visual sin necesidad de agregar nuevas imágenes. Las transformaciones incluyen:

- **Giro horizontal aleatorio** (`RandomFlip`)
- **Rotación aleatoria** (`RandomRotation`)
- **Zoom aleatorio** (`RandomZoom`)

Estas técnicas ayudan a reducir el sobreajuste y permiten que el modelo aprenda características más generales del contenido visual.

![image](https://github.com/user-attachments/assets/65bc234d-d7e0-4083-ab72-3a0ad6e2556b)

## 🤖 Construcción del modelo

### Arquitectura CNN
Una **Red Neuronal Convolucional (CNN)** es una arquitectura de red para Deep Learning que aprende directamente a partir de datos. Mediante capas de convolución aprende filtros que, de forma jerárquica, reconocen desde bordes y texturas hasta composiciones visuales complejas; las capas de *max-pooling* comprimen la información, otorgando invarianza a traslaciones y escalas. Así, la CNN elimina la dependencia de descriptores manuales y se adapta a la gran variabilidad presente en las capturas de videojuegos.

![image](https://github.com/user-attachments/assets/eae4b4b3-83bc-462d-9d12-417ec1106cd6)

#### ¿Por qué una CNN y no SVM / Random Forest?
- **Extracción de características automática.** La red aprende sola los filtros que distinguen *pixel-art* (Minecraft) de texturas realistas (CS:GO), algo que los enfoques basados en HOG + SVM no logran sin una ingeniería manual costosa.  
- **Robustez a transformaciones.** Convolución + pooling mantiene el rendimiento ante cambios de brillo, giros leves o escalados típicos de un *screenshot*.  
- **Escalabilidad.** La arquitectura puede crecer (más filtros/capas) o migrar, mientras que los modelos clásicos requieren redefinir descriptores.  

> **Paper de Referencia**. Breve F. (2023) *From Pixels to Titles: Video Game Identification by Screenshots using Convolutional Neural Networks*, arXiv:2311.15963

### Composición de la CNN

| Bloque | Capa | Parámetros clave | Propósito |
|--------|------|------------------|-----------|
| 1 | `Conv2D(32, 3×3)` + ReLU | Entrada `224×224×3` | Detecta bordes/texturas básicas |
|   | `MaxPooling2D(2×2)` | — | Reduce resolución, gana invarianza |
| 2 | `Conv2D(64, 3×3)` + ReLU | — | Captura patrones intermedios |
|   | `MaxPooling2D(2×2)` | — | — |
| 3 | `Conv2D(128, 3×3)` + ReLU | — | Reconoce estructuras complejas (HUD, mapas) |
|   | `MaxPooling2D(2×2)` | — | — |
| — | `Flatten()` | — | Pasa de 3-D a 1-D |
| — | `Dense(128)` + ReLU | — | Combina rasgos globales |
| — | `Dense(6)` + Softmax | — | Probabilidades para las 6 clases |

### Selección de métricas

| Métrica | Qué mide | Por qué es útil |
|---------|----------|-----------------|
| **Accuracy** | Proporción de aciertos globales | Indicador general de desempeño |
| **Precisión** | TP / (TP + FP) por clase | Fiabilidad cuando el modelo “afirma” ser un juego X |
| **Recall** | TP / (TP + FN) por clase | Cuántas imágenes reales de un juego X recupera |
| **F1-score (macro)** | Media balanceada de precisión y recall | Resume errores clase a clase |
| **Support** | Nº de muestras por clase | Conteo de ejemplos reales que pertenecen a cada clase |

Además se genera una **matriz de confusión** para visualizar rápidamente en qué títulos la red se equivoca 
![image](https://github.com/user-attachments/assets/f5dcda05-71de-4d6a-a1dc-14b5de5f12ed)


### Compilación y entrenamiento

| Elemento | Configuración | Razonamiento |
|----------|--------------|--------------|
| **Optimizador** | Adam | Convergencia estable y sin ajuste manual del *learning rate* |
| **Pérdida** | `categorical_crossentropy` | Adecuada para clasificación *one-hot* |
| **Métrica primaria** | `accuracy` | Feedback rápido en cada epoch |

### 🔧 Configuración de entrenamiento

| Hiperparámetro | Valor usado | Por qué se eligió |
|----------------|------------|-------------------|
| **Batch size** | **32** | Con 32 imágenes por lote se logra suficiente aleatoriedad para que el gradiente sea representativo y al mismo tiempo, el consumo de VRAM sea moderado |
| **Épocas** | **10** | Se realizó una primera corrida de 10 pasadas completas sobre los datos. Las curvas *loss/accuracy* mostraron estabilización y ningún signo de overfitting prematuro. |
| **Steps per epoch** | **126** | Se revisan 126 imagenes por época ( 4 006 imágenes de entrenamiento / batch size). Esto permite que cada imagen sea procesada exactamente una vez por época. |
| **Validation steps** | **14** | El conjunto de validación quedó en 445 imágenes → `ceil(445 / 32) = 14` lotes. Suficiente para estimar la generalización sin alargar demasiado cada época. |
| **Pesos de clase** | **No aplicados** | El dataset resultó equilibrado (≈ 700 ± 70 imágenes por clase). Por ello no fue necesario ponderar la pérdida; todas las clases contribuyen por igual durante el aprendizaje. |


