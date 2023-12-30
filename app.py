import matplotlib.pyplot as plt
import numpy as np
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
import cv2

PATH_TO_LABELS = 'data/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def pil_image_as_numpy_array(pilimg):
    img_array = tf.keras.utils.img_to_array(pilimg)
    return img_array

def load_model():
    model_dir = 'saved_model'
    detection_model = tf.saved_model.load(str(model_dir))
    return detection_model

def predict(image_np):
    image_np = pil_image_as_numpy_array(image_np)
    image_np = np.expand_dims(image_np, axis=0)
    results = detection_model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
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
        line_thickness=2
    )
    result_pil_img = tf.keras.utils.array_to_img(image_np_with_detections[0])
    return result_pil_img

def predict_on_video(video_in_filepath, video_out_filepath, detection_model, category_index):
    video_reader = cv2.VideoCapture(video_in_filepath)
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(
        video_out_filepath,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_w, frame_h)
    )
    label_id_offset = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break  # Break the loop if the video is finished
        processed_frame = predict(frame)
        processed_frame_np = np.array(processed_frame)
        video_writer.write(processed_frame_np)
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# Function to process a video
def process_video(video_path):
    output_path = "output_video.mp4"  # Output path for the processed video
    predict_on_video(video_path, output_path, detection_model, category_index)
    return output_path


detection_model = load_model()

# Specify paths to example images
sample_images = [
    ["test_1.jpg"], ["test_9.jpg"], ["test_6.jpg"],
    ["test_7.jpg"], ["test_10.jpg"], ["test_11.jpg"], ["test_8.jpg"]
]

tab1 = gr.Interface(
    fn=predict,
    inputs=gr.Image(label='Upload an expressway image', type="pil"),
    outputs=gr.Image(type="pil"),
    title='Image Processing',
    examples=sample_images
)

# Create the video processing interface
tab2 = gr.Interface(
    fn=process_video,
    inputs=gr.File(label="Upload a video"),
    outputs=gr.File(label="output"),
    title='Video Processing',
    examples=["example_video.mp4"]
)

# Create a Multi Interface with Tabs
iface = gr.TabbedInterface([tab1, tab2], title='Blue and Yellow Taxi detection in live expressway traffic conditions (data.gov.sg)')

# Launch the interface
iface.launch(share=True)