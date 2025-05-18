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

Esta reducción tuvo como objetivo priorizar la calidad y la diversidad del contenido, manteniendo un balance entre clases y una proporción de **80% para entrenamiento y 20% para prueba**.

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
