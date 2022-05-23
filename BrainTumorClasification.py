from gc import callbacks
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os
import sys 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



#Para utilizar el proyecto y ejecutar el modelo, es nencesario irse al siguiente link, descargar la base de datos y colocarla en esta carpeta: 
#https://www.kaggle.com/code/purvitsharma/brain-tumor-classification-98-4-accuracy/data?select=Brain+Tumor+Data+Set

link_descarga = "https://www.kaggle.com/code/purvitsharma/brain-tumor-classification-98-4-accuracy/data?select=Brain+Tumor+Data+Set"

if(not os.path.exists("'Brain Tumor Data Set.zip'")): 
    sys.exit(f"Es necesario descargar el .zip de la siguiente página: {link_descarga} y mover el .zip a esta carpeta")

# Rutina para darle formato a las bases de datos que se usarán en este sistema 
os.system("unzip 'Brain Tumor Data Set.zip' ")
os.system("mv 'Brain Tumor Data Set' 'Brain_Tumor_Dataset' ")
os.system("mv 'Brain_Tumor_Dataset/Brain Tumor' 'Brain_Tumor_Dataset/Brain_Tumor' ")

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(tf.__version__)

# Se va a crear un dataframe con la información de los directorios
tumor_dir= "./Brain_Tumor_Dataset/Brain_Tumor"
healthy_dir= "./Brain_Tumor_Dataset/Healthy"
filepaths = []
labels= []
dict_list = [tumor_dir, healthy_dir]
for i, j in enumerate(dict_list):
    flist=os.listdir(j)
    for f in flist:
        fpath=os.path.join(j,f)
        filepaths.append(fpath)
        if i==0:
          labels.append('cancer')
        else:
          labels.append('healthy')

Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
tumor_data = pd.concat([Fseries,Lseries], axis=1)
tumor_df = pd.DataFrame(tumor_data)

# Vamos a generar dos conjuntos de datos para trabajar con el modelo
train_images, test_images = train_test_split(tumor_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(tumor_df, test_size=0.2, random_state=42)

# Vamos a generar los conjuntos de datos de entrenamiento, prueba y de validación
image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)

train = image_gen.flow_from_dataframe(dataframe= train_set,x_col="filepaths",y_col="labels",
                                      target_size=(244,244),
                                      color_mode='rgb',
                                      class_mode="categorical", #used for Sequential Model
                                      batch_size=32,
                                      shuffle=False            #do not shuffle data
                                     )
test = image_gen.flow_from_dataframe(dataframe= test_images,x_col="filepaths", y_col="labels",
                                     target_size=(244,244),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=32,
                                     shuffle= False
                                    )
val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="filepaths", y_col="labels",
                                    target_size=(244,244),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=32,
                                    shuffle=False
                                   )

classes=list(train.class_indices.keys())
num_classes = len(classes)
print("Categorías de clasificación: ",classes)

def show_images(image_gen, titulo):
    global classes
    images, labels=next(image_gen)
    plt.figure(figsize=(20,20))
    length = len(labels)
    if length<25:
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5,5,i+1)
        image=(images[i]+1)/2 # Factor escala
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color="green",fontsize=16)
        plt.axis('off')
    plt.suptitle(titulo, fontsize = 18)
    plt.show()

show_images(train, "Conjunto de Entrenamiento")

img_height = 244
img_width  = 244
epochs = 10

# Ahorita vamos a prescindir de esta condición para que se puedan visualizar los resultados en el servidor local
if(1 == 2):#os.path.exists("prediccion_tumor_cerebral.h5")):
    model = keras.models.load_model("prediccion_tumor_cerebral.h5")
    history = model.history
else:
    # Modelo de Predicción:
    model = Sequential([
        layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1), activation="relu", padding="valid",
                input_shape=(244,244,3)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(rate = 0.3),
        layers.Dense(64, activation='relu'),
        # Vamos a agregar 2 layers más a la arquitectura de la red
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Vamos a imprimir el resumen del modelo
model.summary()

# Vamos a visualizar los resultados en un servidor local
root_logdir = os.path.join(os.curdir, "logs")
# Función para obtener el tiempo
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_dir = get_run_logdir(root_logdir)

tensorboard_cb = keras.callbacks.TensorBoard(run_dir)

# Vamos a uterar con 1p epochs
history = model.fit(train, validation_data= val, epochs=epochs,verbose=1, callbacks=[tensorboard_cb])
# Escribir el siguiente comando en la consola:
# tensorboard --logdir ./logs --port=6006

# Vamos a mostrar los resultados del entrenamiento:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Vamos a guardar el modelo:
model.save("prediccion_tumor_cerebral.h5")

# Se ejecuta la visualización de datos desde un servidor local
os.system("tensorboard --logdir logs --port=6006")
