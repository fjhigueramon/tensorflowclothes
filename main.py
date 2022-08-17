import tensorflow as tf
import tensorflow_datasets as tfds
import math

# Variables globales
tamanoLotes = 32


# Metodo de normalizacion (0-255) --> (0-1)
def normalize(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas


# Recogemos el set de datos de las imagenes de ropa de Zalando
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# print(metadatos)
# Aqui podemos ver que hay 60000 ejemplos para entrenar y 10000 ejemplos para probar

# Separamos los 2 conjuntos de ejemplos en datos para entrenar y datos para test
datosEntrenamiento, datosPrueba = datos['train'], datos['test']

# Obtener los nombres de las categorias
nombres_clases = metadatos.features['label'].names

# print(nombres_clases)

# Normalizacion de los datos
datosEntrenamiento = datosEntrenamiento.map(normalize)
datosPrueba = datosPrueba.map(normalize)

# Agregar a cache (Usar memoria en vez de disco)
datosEntrenamiento = datosEntrenamiento.cache()
datosPrueba = datosPrueba.cache()

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compilacion del modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

numEjemplosEntrenamiento = metadatos.splits["train"].num_examples
numEjemplosPrueba = metadatos.splits["test"].num_examples

datosEntrenamiento = datosEntrenamiento.repeat().shuffle(numEjemplosEntrenamiento).batch(tamanoLotes)
datosPrueba = datosPrueba.batch(tamanoLotes)

# Entrenamiento
# Entrenamos el modelo con 20 etapas --> Después de varias pruebas se concluye que aumentar el número de etapas no
# proporciona mejoras sustanciales
historial = modelo.fit(datosEntrenamiento, epochs=20, steps_per_epoch=math.ceil(numEjemplosEntrenamiento/tamanoLotes))
