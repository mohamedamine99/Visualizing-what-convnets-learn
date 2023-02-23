# -*- coding: utf-8 -*-
"""
Contains functions for visualizing intermediate convnet outputs (intermediate activations), 
convnet filters, and heatmaps of class activation in an image of the CNN model.

"""

# file and directories manipulation
import os
import shutil

# loading and building new models
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import models

# image data manipulation
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_model_activations(img_path, model, verbose=False):
    """
    Get the intermediate activations of the input image using the provided CNN model.

    Args:
        img_path (str): The path to the input image file.
        model (tf.keras.Model): The CNN model used to extract the activations.
        verbose (bool): If True, display the shape of each layer's activations. Defaults to False.

    Returns:
        List of numpy arrays: A list containing the activations of each layer in the CNN model.

    Raises:
        ValueError: If the input image path is not a string.
        TypeError: If the input model is not a tensorflow.keras.Model object.

    Examples:
        >>> model = tf.keras.applications.VGG16()
        >>> activations = get_model_activations("path/to/image.jpg", model, verbose=True)
        layer n°0 shape : (1, 128, 128, 64)
        layer n°1 shape : (1, 64, 64, 64)
        layer n°2 shape : (1, 64, 64, 128)
        ...
    """
    if not isinstance(img_path, str):
        raise ValueError("Input image path must be a string.")
        
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Input model must be a tensorflow.keras.Model object.")
    
    # Preprocess the input image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = np.reshape(img, (1, 128, 128, 3))

    # Feed forward the preprocessed image to the activations model
    activations = model.predict(img)

    # Display layer shapes
    if verbose:
        print(len(activations))
        for i, activation in enumerate(activations):
            print(f"Layer n°{i} shape: {activation.shape}")
    
    return activations

def display_activations(img_pth ,activations_model , 
                        images_per_row = 16 , save_dir =None
                       ):
    """Displays and optionally saves the activations of each layer in the model for a given image.

    Args:
        img_pth (str): Path to the image file.
        activations_model (keras.Model): A trained model to compute activations.
        images_per_row (int, optional): Number of images to display in a row. Defaults to 16.
        save_dir (str, optional): Path to the directory where to save the images. If None, images are not saved. 
                                  Defaults to None.

    Returns:
        None.

    Raises:
        ValueError: If the directory specified in save_dir does not exist and cannot be created.

    Example:
        display_activations('path/to/image.jpg', model, images_per_row=8, save_dir='path/to/save/directory')
    """
    # get the list of activations for an image for each layer of the model
    activations = get_model_activations(img_pth
                                        ,activations_model)
    # Get the index of the flatten layer 
    for i, layer in enumerate(model.layers):        
        if layer.name.startswith("flatten"):
            break
    # create a list of model layer names
    layer_names = []
    for idx,layer in enumerate(activations_model.layers[1:i+1]):
        layer_names.append(f'{(idx +1):02d} {layer.name}')

    # create a grid for each layer, and consisting of the layer activations/feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                :, :,
                col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
        plt.axis('off')

        # display the output grid for each layer
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

        # save the grid for each layer in save_dir directory as .png format
        if save_dir is not None:
            if not os.path.isdir(save_dir):
                print(f"Creating {save_dir} ... ")
                os.makedirs(save_dir)
            
            plt.savefig(f'{save_dir}/{layer_name}.png')

# centers images of size (1600,1600) inside a canvas of the same size, used on the generated activations grids

def center_imgs(imgs_dir):
    """
    Center the images in the input directory and add a filename label at the top.

    Args:
        imgs_dir (str): The path to the directory containing the input images.

    Returns:
        None

    Raises:
        TypeError: If the input directory path is not a string.

    Examples:
        >>> center_imgs("path/to/images/")
    """
    if not isinstance(imgs_dir, str):
        raise TypeError("Input directory path must be a string.")
        
    # Loop through each file in the directory
    for i, filename in enumerate(os.listdir(imgs_dir)):
        # Create a blank white image
        img = 255 * np.ones((1600,1600,3))
        
        # Load the image to be centered
        fig = cv2.imread(os.path.join(imgs_dir, filename))
        
        # Center the image within the blank white image
        img[(img.shape[0] // 2 ) - fig.shape[0] // 2:  (img.shape[0] // 2 ) + fig.shape[0] // 2] = fig
        
        # Add the filename as a label at the top of the image
        text = f'{filename[:-4]}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 2.75
        text_thickness = 8
        text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
        height, width, _ = img.shape
        
        x = int((width - text_size[0]) / 2)
        
        img = cv2.putText(img,text,(x,150),font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        
        # Save the centered image with the filename label
        cv2.imwrite(f"{imgs_dir}/{filename}", img)

def create_gif(source_path: str, save_dir: str, fps: float = 0.5) -> None:
    """
    Creates a GIF file from a sequence of images located within a directory.

    Args:
        source_path (str): The path to the directory containing the images.
        save_dir (str): The directory to save the resulting GIF file to.
        fps (float, optional): The number of frames per second of the resulting GIF file. Default is 0.5.

    Returns:
        None
    """
    with imageio.get_writer(save_dir, mode='I', fps=fps) as writer:
        for filename in sorted(os.listdir(source_path)):
            image = imageio.imread(os.path.join(source_path, filename))
            writer.append_data(image)

# adjusts a text in an image
def center_text(img, text= '', font_scale = 0.5, offset = (0, 0)) :
    """
    Adjusts the position of text in an image to center it horizontally.

    Args:
        img (numpy.ndarray): The input image.
        text (str, optional): The text to center. Default is an empty string.
        font_scale (float, optional): The size of the font to use. Default is 0.5.
        offset (Tuple[int, int], optional): The (x,y) offset to add to the center position. Default is (0,0).

    Returns:
        numpy.ndarray: The modified image with the centered text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the size of the text string
    (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness=2)[0]

    # Calculate the x, y coordinates for the text
    x = int((img.shape[1] - text_width) / 2)
    y = text_height + 5

    img = cv2.putText(img, text, (x + offset[0], y + offset[1]), font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
    
    return img

# save each activation for each layer individually of a certain img
def save_conv_activations(img_path, model ,save_dir ):
    
    """Save the convolutional activations of a given image for each feature map.

    Args:
        img_path (str): The path to the input image.
        model (keras.models.Model): The model from which to get activations.
        save_dir (str): The directory in which to save the activations.

    Returns:
        None
    """

    # save only the conv layers indices in a list

    conv_idxs = [i for i,layer in enumerate(model.layers) if "conv" in layer.name]
    print(conv_idxs)
    activations = get_model_activations(img_path,model)
    print(conv_idxs)
    for layer in conv_idxs:
        for feature_map in range(activations[layer].shape[3]):
            activation = activations[layer][0,:,:,feature_map]

            # Normalize the grayscale image
            normalized = cv2.normalize(activation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply viridis colormap
            viridis = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
            im_size = viridis.shape[0]
            viridis = cv2.resize(viridis,(256,256))


            # border to add to the image in order to expand it
            border_width = 75
            border_color = [255, 255, 255]  # white color in BGR format

            # Add border to the image
            viridis = cv2.copyMakeBorder(viridis, border_width, border_width, 
                                         border_width, border_width, cv2.BORDER_CONSTANT, 
                                         value=border_color)

            # Get the size of the text string
            text1 = "Size -- Layer idx -- Feature map idx"
            text2=f'({im_size}x{im_size})--{layer:02d}--{feature_map:03d}'

            # add adjusted text
            center_text(viridis,text1)
            center_text(viridis,text2,font_scale = 1 ,offset = (0,40))
        
            # Save the centered image
            cv2.imwrite(f'{save_dir}/{layer:02d}--{feature_map:03d}.png',viridis)
