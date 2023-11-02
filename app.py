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
import PIL
import gradio as gr

PATH_TO_LABELS = 'data/label_map.pbtxt'   
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def pil_image_as_numpy_array(pilimg):

    (im_width, im_height) = pilimg.size
    
    return np.array(pilimg.getdata())[:,:3].reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)
    

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    return pil_image_as_numpy_array(image)            

def load_model():
    wget.download("https://nyp-aicourse.s3-ap-southeast-1.amazonaws.com/pretrained-models/balloon_model.tar.gz")
    tarfile.open("balloon_model.tar.gz").extractall()
    model_dir = 'saved_model'    
    detection_model = tf.saved_model.load(str(model_dir))
    return detection_model    

# samples_folder = 'test_samples
# image_path = 'test_samples/sample_balloon.jpeg
# 

def predict(pilimg):

    image_np = pil_image_as_numpy_array(pilimg)
    return predict2(image_np)

def predict2(image_np):

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

detection_model = load_model()
# pil_image = Image.open(image_path)
# image_arr = pil_image_as_numpy_array(pil_image)

# predicted_img = predict(image_arr)
# predicted_img.save('predicted.jpg')

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Image(type="pil")
             ).launch(share=True)
# def greet(name):
    # return "Hello " + name + "!!"
# 
# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch()
    # fn=predict,
    # inputs=["image"],
    # outputs=[gr.outputs.Image(type="pil", label="RetinaNet Inference")],
    # title="Fridge Object Detector",
    # description="A VFNet model that detects common objects found in fridge. Upload an image or click an example image below to use.",
# )
# 