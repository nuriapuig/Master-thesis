import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import os 
from keras import backend as K
import cv2
import matplotlib.cm as cm
from IPython.display import Image, display

def load_data ():
    
    lung_aca = '../lung_image_sets/lung_aca'
    lung_n = '../lung_image_sets/lung_n'
    lung_scc = '../lung_image_sets/lung_scc'
    
    return lung_aca, lung_n, lung_scc

def vis_data (lung_aca, lung_n, lung_scc):

    #Muestro por pantalla 4 ejemplos de Adenocarcinoma
    
    fig, axis = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Lung adenocarcinoma')
    
    axis[0,0].imshow(load_img(lung_aca + "/lungaca1.jpeg"))
    axis[0,0].axis("off")
    axis[0,1].imshow(load_img(lung_aca + "/lungaca2.jpeg"))
    axis[0,1].axis("off")
    axis[1,0].imshow(load_img(lung_aca + "/lungaca3.jpeg"))                   
    axis[1,0].axis("off")
    axis[1,1].imshow(load_img(lung_aca + "/lungaca4.jpeg"))
    axis[1,1].axis("off")
    
    plt.show()


    #Muestro por pantalla 4 ejemplos de Benign Tissue
    fig, axis = plt.subplots(2,2, figsize = (8,8))
    fig.suptitle('Lung benign tissue')
    
    axis[0,0].imshow(load_img(lung_n + '/lungn1.jpeg'))
    axis[0,0].axis('off')
    axis[0,1].imshow(load_img(lung_n + '/lungn2.jpeg'))
    axis[0,1].axis('off')
    axis[1,0].imshow(load_img(lung_n + '/lungn3.jpeg'))
    axis[1,0].axis('off')
    axis[1,1].imshow(load_img(lung_n + '/lungn4.jpeg'))
    axis[1,1].axis('off')

    plt.show()
    
    #Muestro por pantalla 4 ejemplos de Squamous Cell Carcinoma
    fig, axis = plt.subplots(2,2, figsize = (8,8))
    fig.suptitle('Lung squamous cell carcinoma')
    
    axis[0,0].imshow(load_img(lung_scc + '/lungscc1.jpeg'))
    axis[0,0].axis('off')
    axis[0,1].imshow(load_img(lung_scc + '/lungscc2.jpeg'))
    axis[0,1].axis('off')
    axis[1,0].imshow(load_img(lung_scc + '/lungscc3.jpeg'))
    axis[1,0].axis('off')
    axis[1,1].imshow(load_img(lung_scc + '/lungscc4.jpeg'))
    axis[1,1].axis('off')

    plt.show()
    
    return None

def general_preprocessing(image_path, label):
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, (350, 350))  
    image = tf.cast(image, tf.float32)

    return image, label

def get_dataset(preprocess, lung_aca, lung_n, lung_scc):

    lung_aca_paths = []
    lung_n_paths = []
    lung_scc_paths = []
    
    for root, dirs, files in os.walk(lung_aca, topdown=False): #root, dirs y files son las variables que siempre devuelve os.walk. 
        for name in files:                                     
            lung_aca_paths.append(os.path.join(root, name))

    for root, dirs, files in os.walk(lung_n, topdown=False):
        for name in files:
            lung_n_paths.append(os.path.join(root, name))
    
    for root, dirs, files in os.walk(lung_scc, topdown=False):
        for name in files:
            lung_scc_paths.append(os.path.join(root, name))

    labels = [5000 * ['Adenocarcinoma'], 5000 * ['BenignTissue'], 5000 * ['SquamousCellCarcinoma']]
    labels = np.concatenate(labels, axis = 0) #Con esto conseguimos que todo esté en un mismo axis
    image_paths = lung_aca_paths + lung_n_paths + lung_scc_paths

    encoder = OneHotEncoder(sparse=False)

    labels = labels.reshape(-1, 1)
    labels = encoder.fit_transform(labels)

    train_pth, test_pth, train_lbl, test_lbl = train_test_split(image_paths, labels, test_size = 1500, stratify = labels, random_state = 42)
    train_pth, val_pth, train_lbl, val_lbl = train_test_split(train_pth, train_lbl, test_size=1500, stratify = train_lbl,random_state=42)
    
    #Creamos los datasets, usando el preprocesado que se ha tomado como argumento
    train_dataset = tf.data.Dataset.from_tensor_slices((train_pth, train_lbl))
    train_dataset = (
        train_dataset.map(preprocess)
        .batch(32)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_pth, val_lbl))
    val_dataset = (
        val_dataset.map(preprocess)
        .batch(32)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_pth, test_lbl))
    test_dataset = (
        test_dataset.map(preprocess)
        .batch(32)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    return train_dataset, val_dataset, test_dataset

def get_results(test_dataset, y_pred):
    
    # Initialize lists to store extracted x and y
    all_y = []

    # Iterate through batches and extract x and y
    for x_batch, y_batch in test_dataset:
        all_y.append(y_batch)
    
        # Break the loop if you have extracted all the data you need
        # For example, if you want a specific number of batches, you can set a counter and break when reached

    # Concatenate the lists to get the complete dataset
    y_test = np.concatenate(all_y, axis=0)

    argmax_pred = np.argmax(y_pred, axis = 1)

    y_pred_arg = np.zeros_like(y_pred)
    y_pred_arg[np.arange(len(y_pred)), argmax_pred] = 1
    y_pred_arg.shape

    y_test.shape

    results = classification_report(y_test, y_pred_arg)
    
    return results

def draw_heatmap(test_dataset, y_pred):
    
    # Initialize lists to store extracted x and y
    all_y = []

    # Iterate through batches and extract x and y
    for x_batch, y_batch in test_dataset:
        all_y.append(y_batch)
    
        # Break the loop if you have extracted all the data you need
        # For example, if you want a specific number of batches, you can set a counter and break when reached

    # Concatenate the lists to get the complete dataset
    y_test = np.concatenate(all_y, axis=0)

    argmax_pred = np.argmax(y_pred, axis = 1)

    y_pred_arg = np.zeros_like(y_pred)
    y_pred_arg[np.arange(len(y_pred)), argmax_pred] = 1
    y_pred_arg.shape

    y_pred_arg = np.argmax(y_pred_arg, axis=1)
    y_pred_arg.shape
    y_test = np.argmax(y_test, axis=1)
    confusion = confusion_matrix(y_test, y_pred_arg)

    return confusion

def plot_maps(img1, img2,sal_path, vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(15,45))
    plt.subplot(1,3,1)
    plt.imshow(img1,vmin=vmin, vmax=vmax, cmap="magma")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap="magma")
    plt.axis("off")
    ax = plt.subplot(1,3,3)
    plt.imshow(img1*mix_val+img2/mix_val, cmap = "magma" )
    extent = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    ax.figure.savefig(sal_path,  bbox_inches=extent)
    plt.axis("off")

def normalize_image(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm

def draw_saliency(image_path, model, pre_fn, sal_path):

    img , _ = pre_fn(image_path,0)
    img = tf.expand_dims(img, axis = 0)

    with tf.GradientTape() as tape:
        tape.watch(img)
        result = model(img)
        max_idx = tf.argmax(result,axis = 1)
        max_score = result[0,max_idx[0]]
        grads = tape.gradient(max_score, img)

    plot_maps(normalize_image(grads[0]), 
    normalize_image(img[0]), sal_path)

def draw_CAM(img_path, size, model, model_key, cam_path, alpha, pre_fn, pred_index = None):
    # Esta función está basada en el código de documentación de Keras correspondiente a grad_cam, consultable en: https://keras.io/examples/vision/grad_cam/
    # Chollet, F.  Grad-CAM class activation visualization. (7 de marzo de 2021). Keras. https://keras.io/examples/vision/grad_cam/
    
    last_conv_index = {
    "simpleConv": -29,
    "efficient": -26,
    "inception": -10,
    "resnet": -8,
    }

    img , _ = pre_fn(img_path,0)
    
    img = tf.expand_dims(img, axis = 0)
    img_array = img

    grad_model = keras.models.Model(
        model.inputs, [model.layers[last_conv_index[model_key]].output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


