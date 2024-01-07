# APLICACIÓN DE REDES NEURONALES CONVOLUCIONALES PARA EL DIAGNÓSTICO DE CÁNCER DE PULMÓN A PARTIR DE IMÁGENES HISTOLÓGICAS

## Descripción del proyecto

El proyecto se centra en la aplicación de redes neuronales convolucionales para la clasificación de imágenes histológicas de cáncer de pulmón según si corresponden a tejido benigno, a un adenocarcinoma, o a un carcinoma de células escamosas. Dos de las redes aplicadas se han diseñado desde cero, mientras que las otras dos son grandes redes neuronales aplicadas mediante *Transfer Learning*.

Además, también implementa *Saliency Maps* y *Class Activation Maps* para visualizar la activación neuronal de distintas redes y poder evaluar su explicabilidad.

El procedimiento de desarrollo se detalla ampliamente en la memoria adjunta, en la que también se presentan y discuten los resultados, se analizan aspectos clave para una hipotética implementación como producto de IA y se presentan las posibles consecuencias de la misma.

## Instrucciones

### Obtención de datos

Las imágenes usadas como input forman parte del dataset *Lung and Colon Cancer Histopathological Images*, y se han descargado a partir del siguiente [enlace a Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

En el proyecto se ha trabajado únicamente con la parte del dataset correspondiente a imágenes pulmonares, por lo que únicamente es necesario descargar la carpeta *lung_image_sets* desde el ya mencionado enlace a Kaggle. Esta carpeta se debe descargar y guardar en la raíz del proyecto.

Dentro de la carpeta *lung_image_sets* se encuentran tres carpetas más, denominadas *lung_aca*, *lung_n* y *lung_scc*. Estas tres carpetas corresponden a las tres clases en las que se clasifican las imágenes: adenocarcinoma, tejido benigno y carcinoma de células escamosas, respectivamente. En cada una de las carpetas se encuentran las 5000 imágenes correspondientes a dicha clase, sumando el total de 15000 imágenes de tejido pulmonar que figuran en el dataset.

### Instalación y ejecución

Para poder ejecutar las *notebooks* del proyecto, es necesario instalar las librerías que se incluyen en el documento *requirements*, en la versión adecuada. Para ello, se recomienda crear un *virtual environment* en el que instalar dichas librerías y ejecutar posteriormente el código del proyecto.

Para crear un *virtual environment* para proyectos de Python, hay que abrir una nueva ventana en la terminal y ejecutar el siguiente comando:

```bash
python -m venv venv
```

Al hacerlo, se ha creado un *virtual environment* llamado venv. A continuación, es necesario activar el *virtual environment* que se ha creado con el objetivo de que las ejecuciones se den en este environment y tenga acceso a las librerías básicas de Python.

Para llevar a cabo dicha activación, se ha de ejecutar en la terminal (después del comando anterior), el siguiente comando en caso de usar Linux:

```bash
source venv/bin/activate
```

O el siguiente, en caso de usar Windows:

```bash
source ./venv/bin/Activate.pst
```

Una vez activado el *virtual environment*, ya es posible instalar las librerías usadas en el proyecto, que están detalladas en el documento *requirements*, ejecutando el siguiente comando en la terminal:

```bash
pip install -r requirements.txt
```

## Estructura del proyecto

```
.
├───Core
│   ├───CamMaps # Contiene todos los Class Activation Maps
│   │
│   ├───Results # Contiene las matrices de Saliency Maps y CAMs que se presentan como resultados.
│   │
│   └───SalMaps # Contiene todos los Saliency Maps
│   │
│   └───utils.py # Contiene las funciones que se usan en varias notebooks.
│   │
│   └───train*.ipynb # 4 notebooks, cada una con el entrenamiento y la generación de mapas de una de las redes.
│   │                
│   └───Visualization.ipynb # Notebook para la generación de las matrices de mapas
│ 
└───ModelCheckpoints # Contiene los checkpoints de las redes aplicadas. 
                     # En las redes que se aplican mediante Transfer Learning, hay un checkpoint
                     # para el primer entrenamiento y otro para el Fine Tuning.

```

