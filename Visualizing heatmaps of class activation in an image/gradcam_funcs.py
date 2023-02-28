
# testing and visualizing CNN model predictions
def test_CNN_model(CNN_model= None, test_imgs_path = '', labels = []):
    pre_processed_imgs = []
    ncols = len(os.listdir(test_imgs_path))
    fig = plt.figure(figsize=(8, 5))
    for i , img_file in enumerate(os.listdir(test_imgs_path)):
        print(img_file)
        img = cv2.imread(os.path.join(test_imgs_path,img_file))
        if (img is None):
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(1, ncols, i+1)
        plt.imshow(img)
        plt.axis('off')
        
        img = cv2.resize(img,(128,128))
        img = img / 255
        img = np.reshape(img,(1,128,128,3))
        pre_processed_imgs.append(img)
        results = CNN_model.predict(img,verbose = 0)
        print(results)
        results = np.squeeze(results)
        plt.title(labels[np.round(results).astype(int)])

        print(results)
    return pre_processed_imgs


def preprocess_img(img : np.ndarray , size = (128,128)):
    img = cv2.resize(img, size)
    img = img / 255
    img = np.expand_dims(img, axis = 0)
    return img

def extract_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            #print(layer.output_shape)
            return layer.name

def generate_heatmap(img : np.ndarray, model = None, activations_layer ='auto', 
                     class_id = 1 , cmap = cv2.COLORMAP_HOT):

    if activations_layer == 'auto':
        layer_name = extract_last_conv_layer(model)
        last_conv_layer = model.get_layer(layer_name)
    else:
        last_conv_layer = model.get_layer(activations_layer)

    grads_model = Model(inputs = [model.inputs], outputs = [last_conv_layer.output, model.output])
    #print(type(grads_model))
    with tf.GradientTape() as tape:
        img = tf.cast(img, tf.float32)
        last_conv_activations, preds = grads_model(img)

        #print(model.output_shape[1])

        if model.output_shape[1] == 1:
            grads = tape.gradient(preds , last_conv_activations)

        else :
            grads = tape.gradient(preds[:,class_id] , last_conv_activations)
    
    #print(type(grads))
    cast_grads = tf.cast(grads > 0 , tf.float32)
    cast_activations = tf.cast(last_conv_activations > 0 , tf.float32 )
    guided_grads = cast_grads * cast_activations * grads

    #print(guided_grads.shape)

    guided_grads = guided_grads[0]
    last_conv_activations = last_conv_activations[0]

    weights = tf.reduce_mean(guided_grads, axis = (0,1))

    heatmap = tf.multiply(last_conv_activations, weights )
    heatmap = tf.reduce_sum(heatmap , axis = -1)

    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) - np.min(heatmap))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap , (img.shape[2], img.shape[1]))
    heatmap = cv2.applyColorMap(heatmap, cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    #print((img.shape[1], img.shape[0]))
    #plt.imshow(heatmap)

    return heatmap

def overlay_heatmap(heatmap : np.ndarray , img: np.ndarray, alpha = 0.5 , beta = 0.7):
    heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
    output = cv2.addWeighted(img, alpha , heatmap, beta, 0)
    #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    #plt.imshow(output)
    return output


def video_heatmap_overlay(video_file , save_file, 
                        model, activations_layer ='auto',
                        preprocess_size = (128,128),
                        class_id = 10,
                        cmap = cv2.COLORMAP_JET,
                        alpha = 0.5 , beta = 0.7) :

    cap = cv2.VideoCapture(video_file)

    width  = int(cap.get(3) )  # get `width` 
    height = int(cap.get(4) )  # get `height` 

    # define an output VideoWriter  object
    out = cv2.VideoWriter(save_file,
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        15,(width,height))

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error opening video stream or file")


    # Read the video frames
    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error reading frame")
            print((width,height),overlayed.shape)
            break

        # Capture the video frame
        # by frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_processed = preprocess_img(frame , size = preprocess_size)

        heatmap = generate_heatmap(frame_processed, model = model, class_id = class_id,
                                   activations_layer = activations_layer, cmap = cmap)
        overlayed = overlay_heatmap(heatmap , frame , alpha = alpha , beta = beta)

        overlayed = cv2.cvtColor(overlayed , cv2.COLOR_BGR2RGB)

        out.write(overlayed)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    out.release()

def GIF_from_vid(vid_file, gif_file, fps = 20, skip = 2):
    i = 0

    cap = cv2.VideoCapture(vid_file)
    width  = int(cap.get(3) )  # get `width` 
    height = int(cap.get(4) )  # get `height` 

    # Create a writer object to write the frames to a GIF file
    writer = imageio.get_writer(gif_file, mode='I',fps=fps)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error opening video stream or file")


    # Read the video frames
    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error reading frame")
            break
        i+=1
        if( i % skip == 0):
            continue
        # add current frame to the GIF file
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame)

    # Close the reader and writer objects
    writer.close()
    cap.release()

gifs_dir ='/content/gifs'
if not os.path.exists(gifs_dir):
    os.mkdir(gifs_dir)
    print(f"Directory '{gifs_dir}' created!")

for filename in os.listdir(results_path):
    if filename.endswith(".avi"):
        vid_file = os.path.join(results_path, filename)
        gif_file = os.path.join(gifs_dir,filename[:-4] + ".gif")
        
        GIF_from_vid(vid_file, gif_file, fps = 20, skip = 2)