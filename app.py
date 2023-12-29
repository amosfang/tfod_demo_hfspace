import matplotlib.pyplot as plt
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_op
import tarfile
import wget 
import gradio as gr
from huggingface_hub import snapshot_download
import os 

PATH_TO_LABELS = 'data/label_map.pbtxt'   
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def pil_image_as_numpy_array(pilimg):

    img_array = tf.keras.utils.img_to_array(pilimg)
    # img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
def load_image_into_numpy_array(path):
                                    
    image = None
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    return pil_image_as_numpy_array(image)            

def load_model():
    model_dir = 'saved_model'    
    detection_model = tf.saved_model.load(str(model_dir))
    return detection_model    


def predict(image_np):
    
    image_np = pil_image_as_numpy_array(image_np)
    image_np = np.expand_dims(image_np, axis=0)
    
    results = detection_model(image_np)

    # different object detection models have additional results
    result = {key:value.numpy() for key,value in results.items()}
    
    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.60,
        agnostic_mode=False,
        line_thickness=2)

    result_pil_img = tf.keras.utils.array_to_img(image_np_with_detections[0])
    
    return result_pil_img

# def predict2(pilimg):
#     image = None
#     image = load_image_into_numpy_array(pilimg)
#     return predict(image)
    
detection_model = load_model()

# Specify paths to example images
sample_images = [["test_1.jpg"],["test_9.jpg"],["test_6.jpg"],["test_7.jpg"],
                 ["test_10.jpg"], ["test_11.jpg"],["test_8.jpg"]]

# Create a list of example inputs and outputs using a for loop
# example_inputs = [Image.open(image) for image in sample_images]
# example_outputs = [predict(input_image) for input_image in example_inputs]

# Save the example output image
# example_outputs[0].save("/home/user/app/predicted_1.jpg")

iface = gr.Interface(fn=predict,
                     inputs=gr.Image(label='Upload an expressway image', type="pil"),
                     outputs=gr.Image(type="pil"),
                     title='Blue and Yellow Taxi detection in live expressway traffic conditions (data.gov.sg)',
                     examples = sample_images
                    )

iface.launch(share=True)