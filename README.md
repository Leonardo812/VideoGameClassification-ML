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

La elección de una red neuronal convolucional (CNN) se justifica en tres frentes, todos documentados en el trabajo de Breve [1]:

- **Extracción de características automática.** El estudio muestra que las CNN obtienen “resultados sobresalientes sin necesidad de describir manualmente las características de cada juego” [1, Sec. I]. En contraste, métodos clásicos como HOG + SVM requieren diseñar descriptores a mano, lo que limita la capacidad de adaptación cuando se añaden títulos con estilos gráficos nuevos. 
- **Robustez a transformaciones.** Las etapas de convolución y *max-pooling* confieren a la red “invariancia natural a traslaciones, rotaciones y cambios de escala”. Esa propiedad es crucial porque las capturas de pantalla llegan con distintas resoluciones, encuadres y niveles de brillo; un SVM o un Random Forest no poseen esa resistencia de forma inherente.  
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


### 🔧 Configuración de entrenamiento

| Elemento | Configuración | Razonamiento |
|----------|--------------|--------------|
| **Optimizador** | Adam | Convergencia estable y sin ajuste manual del *learning rate* |
| **Pérdida** | `categorical_crossentropy` | Adecuada para clasificación *one-hot* |
| **Métrica primaria** | `accuracy` | Feedback rápido en cada epoch |

| Hiperparámetro | Valor usado | Por qué se eligió |
|----------------|------------|-------------------|
| **Batch size** | **32** | Con 32 imágenes por lote se logra suficiente aleatoriedad para que el gradiente sea representativo y al mismo tiempo, el consumo de VRAM sea moderado |
| **Épocas** | **10** | Se realizó una primera corrida de 10 pasadas completas sobre los datos. Las curvas *loss/accuracy* mostraron estabilización y ningún signo de overfitting prematuro. |
| **Steps per epoch** | **126** | Se revisan 126 imagenes por época ( 4 006 imágenes de entrenamiento / batch size). Esto permite que cada imagen sea procesada exactamente una vez por época. |
| **Validation steps** | **14** | El conjunto de validación quedó en 445 imágenes /  batch size. Suficiente para estimar la generalización sin alargar demasiado cada época. |
| **Pesos de clase** | **No aplicados** | El dataset resultó equilibrado ( 700 ± 70 imágenes por clase). Por ello no fue necesario ponderar la pérdida; todas las clases contribuyen por igual durante el aprendizaje. |

## 📈 Evolución del entrenamiento primera version del modelo

![image](https://github.com/user-attachments/assets/2caeae96-fdda-4dc5-9929-ac27283776c8)

![image](https://github.com/user-attachments/assets/9c856dc9-09d9-4a76-b540-ca91af7eebff)

### Análisis de las curvas

- **Crecimiento sólido**. La *accuracy* de entrenamiento pasa de **0.60 → 0.97** en 7 épocas, mientras la de validación sube hasta **0.94**.   
- **Loss estable**. La pérdida de validación oscila entre **0.25 – 0.35**; no hay picos bruscos, señal de que el modelo generaliza con estabilidad.  

> **Conclusión**: el modelo converge rápido y mantiene un equilibrio razonable entre ajuste y generalización; continuar más allá de 10 épocas daría ganancias marginales y riesgo de sobre-ajuste.




![image](https://github.com/user-attachments/assets/9b306a41-80a2-4210-8941-c95def768ae0)

---

## 📈 Desempeño en el conjunto **test**

| Clase            | Precisión | Recall | F1-score | Soporte |
|------------------|-----------|--------|----------|---------|
| Clash-Royale     | **0.98**  | **0.99** | **0.99** | 161 |
| Counter-Strike   | **0.99**  | **1.00** | **1.00** | 154 |
| Dead by Daylight | 0.97 | **1.00** | 0.98 | 157 |
| Minecraft        | 0.98 | 0.79 | 0.88 | 157 |
| Overwatch        | 0.96 | 0.77 | 0.86 | 151 |
| Valorant         | 0.74 | 0.98 | 0.84 | 165 |
| **Accuracy global** |  |  | **0.92** | 945 |
| **Macro avg**    | **0.94** | 0.92 | 0.92 | 945 |
| **Weighted avg** | **0.94** | 0.92 | 0.92 | 945 |

![image](https://github.com/user-attachments/assets/9c618a57-4070-4934-8ce4-5b1910412fe0)

### Analisis de la matriz

* **Clash-Royale, Counter-Strike y Dead by Daylight** se clasifican casi perfectos (> 0.97 F1).  
* **Valorant** presenta la mayor confusión; absorbe muestras de **Minecraft** (25) y **Overwatch** (30), lo que explica su menor precisión (0.74) y el recall alto (0.98).  
* **Minecraft** y **Overwatch** pierden recall por esa misma confusión, aunque mantienen precisión > 0.95.  

### Resumen global

- **Accuracy test = 0.92** confirma que la CNN generaliza bien a imágenes no vistas por lo que el modelo funciona. Sin embargo es muy probable que exitan confusiones que se concentren en los juegos (Valorant, Overwatch yMinecraft-pixel-like). Pues haciendo pruebas de prediccion son los juegos que el modelo confunde mas comunmente. Esto puede ser debido a su similitud visual.

### 🛠️ Refinamiento del modelo

#### Problema detectado  
Aunque la CNN obtenía **92 % de acierto** en el *test set*, fallaba en dos situaciones típicas al clasificar capturas externas:

1. **Iluminación o filtros extremos**  
   Mapas nocturnos, sobreexpuestos o con *shaders* alteraban la paleta y el modelo confundía Valorant ↔ Overwatch o Minecraft ↔ Valorant.
2. **Ausencia de HUD**  
   Algunas capturas sin barra de salud/munición perdían referencias clave; la red asignaba la imagen a la clase visualmente más parecida.

Para corregir estos sesgos se siguieron tres pasos:

Se amplió el set de datos incorporando alrededor de cien capturas adicionales para cada una de las clases problemáticas—Valorant, Overwatch y Minecraft poniendo especial atención en incluir mapas poco frecuentes, escenas sin HUD y material promocional que conserva la paleta y el estilo propios de cada título.

Se depuraron imágenes redundantes o con superposiciones de streaming, de modo que la diversidad efectiva aumentara sin inflar artificialmente la base de datos.

Se añadió regularización adicional activando un Dropout del 20 % en la capa densa final. Esto limita la co-adaptación de neuronas y contrarresta el leve sobre-ajuste observado a partir de la sexta época de entrenamiento original.

Tras la expansión, el conjunto de entrenamiento termino de la siguiente manera:

| Videojuego           | Imágenes (train) |
|----------------------|------------------|
| Dead by Daylight     | 825              |
| Counter-Strike       | 849              |
| Valorant             | 912              |
| Clash Royale         | 718              |
| Minecraft            | 864              |
| Overwatch            | 867              |
| **Total**            | **4,451**        |

*(El set de validación se mantuvo en 10 % de cada clase para conservar la proporción 70-20-10).*

### 📊 Resultados | Comparación antes vs. después del refinamiento

**Desempeño global.**  
El accuracy de prueba pasó de **0.92 a 0.95** y el F1 macro subió de 0.92 a **0.95**. Las curvas de entrenamiento muestran que la diferencia entre train y val sigue por debajo de 5 p.p., por lo que no apareció sobre-ajuste adicional.

## 📈 Desempeño en el conjunto **test**

| Clase            | Precisión | Recall | F1-score | Soporte |
|------------------|-----------|--------|----------|---------|
| Clash-Royale     | **0.89**  | **1.00** | **0.94** | 161 |
| Counter-Strike   | **0.94**  | **1.00** | **0.97** | 154 |
| Dead by Daylight | 0.95 | **1.00** | 0.98 | 157 |
| Minecraft        | 1.00 | 0.88 | 0.94 | 157 |
| Overwatch        | 0.97 | 0.91 | 0.94 | 151 |
| Valorant         | 0.97 | 0.93 | 0.95 | 165 |
| **Accuracy global** |  |  | **0.95** | 945 |
| **Macro avg**    | **0.96** | 0.95 | 0.95 | 945 |
| **Weighted avg** | **0.96** | 0.95 | 0.95 | 945 |

**Cambios por videojuego.**

- **Clash Royale** – El F1 baja de 0.98 a **0.94**. Los trece falsos positivos provenientes de Minecraft sugieren que, al no haberse añadido material nuevo para Clash Royale, algunas de las imágenes incorporadas a las otras clases (en especial Minecraft con shaders y tonalidades cálidas) resultan visualmente parecidas a escenas del juego, generando la confusión.
- **Counter-Strike** – Mantiene un desempeño casi perfecto: F1 pasa de 0.99 a **0.97** (variación mínima).  
- **Dead by Daylight** – Mejora de 0.97 a **0.98**.  
- **Minecraft** – Mejora clara: F1 sube de 0.88 a **0.94** gracias a ejemplos sin HUD y con *shaders*.  
- **Overwatch** – F1 salta de 0.86 a **0.94**; se reduce drásticamente la confusión con Valorant.  
- **Valorant** – Precisión se dispara (0.74 → 0.97) y el F1 llega a **0.95**; el modelo comete muchos menos falsos positivos.

Los gráficos y la nueva matriz de confusión se muestran a continuación:

![image](https://github.com/user-attachments/assets/6328275a-401c-4f0b-a6bf-85624111ff72)


- Las confusiones Valorant → Overwatch caen de 30 a **4**.  
- Minecraft → Valorant desaparece (25 → **0**), aunque parte migra a Clash Royale y Dead by Daylight.  
- Counter-Strike y Dead by Daylight conservan todos sus aciertos en la diagonal.
- Esos errores se reducen a 0 (Minecraft → Valorant) y 4 (Overwatch → Valorant).  
- Aparecen nuevas confusiones menores de Minecraft y Overwatch hacia Clash-Royale (13 y 6 casos) y Dead by Daylight (4 y 3 casos).  

> **Interpretación:** las escenas oscuras o de tonos cálidos que antes se confundían con Valorant ahora se reparten entre Clash-Royale y Dead by Daylight, lo que explica la ligera caída de F1 en Clash-Royale.


![image](https://github.com/user-attachments/assets/17350ceb-59fc-4fd5-a34d-c160673a6cf9)

![image](https://github.com/user-attachments/assets/91fed021-e024-4de0-b200-fe9e329ce531)




### Imagenes de comparación de predicciones

### Modelo previo sin refinar

![image](https://github.com/user-attachments/assets/e76a878b-e12d-47f0-abc1-3088872d5f69)

![image](https://github.com/user-attachments/assets/69cbbd88-ec00-4356-bdb6-fb90725a55e1)

![image](https://github.com/user-attachments/assets/369fd935-a05c-4083-a911-8af2c4752f89)

![image](https://github.com/user-attachments/assets/fa50ccb1-7c63-4f1e-98a7-2fdc8556cc74)

![image](https://github.com/user-attachments/assets/60fcb41f-7dbe-4974-8baf-37331cc41b20)

![image](https://github.com/user-attachments/assets/d22c945f-d025-4797-9660-739c84d8e12a)

![image](https://github.com/user-attachments/assets/b3660e4a-72f3-4e37-9222-bdcd8021e580)





### Modelo refinado

![image](https://github.com/user-attachments/assets/f46070cc-fe65-4281-a35c-1bb1f71acf24)

![image](https://github.com/user-attachments/assets/a8794a13-642f-495d-8f02-60c5ef752342)

![image](https://github.com/user-attachments/assets/72f2e5bb-84f9-4ddc-8a72-fc53e2c282d2)

![image](https://github.com/user-attachments/assets/875a13a8-4d9d-4adc-af10-a4613bf87d25)

![image](https://github.com/user-attachments/assets/b271feb6-7f32-4122-bc7e-7600adae475f)

![image](https://github.com/user-attachments/assets/f51991a0-e478-48cd-930d-8197593ca025)

![image](https://github.com/user-attachments/assets/832bd11f-b95c-4f66-b710-6b65e2af2b75)




> **Conclusión.** Reforzar la variedad del conjunto de datos enfocado en mapas raros, luminación variada y capturas sin HUD y aplicar Dropout bastó para recuperar once puntos porcentuales de precisión en escenarios reales, sin necesidad de cambiar la arquitectura principal. La única oportunidad clara de mejora queda en **Clash Royale**, donde convendría añadir escenas diurnas con filtros cartoon para diferenciarlo de los *shaders* de Minecraft.









