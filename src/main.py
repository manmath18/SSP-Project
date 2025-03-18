import json
from io import BytesIO
from PIL import Image
import os
import tempfile

import boto3
from botocore import UNSIGNED
from botocore.client import Config

import streamlit as st
import pandas as pd
import numpy as np

from resnet_model import ResnetModel
from ultralytics import YOLO

import cv2
import matplotlib.pyplot as plt

from dashboard import display_dashboard  # Import the dashboard component

def load_model(path: str = "utils/runs/detect/train2/weights/best4.pt") -> ResnetModel:
    model = YOLO(path, "v8")
    return model

def load_index_to_label_dict(path: str = "utils/class_label.json") -> dict:
    with open(path, "r") as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()
    }
    return index_to_class_label_dict

def load_files_from_s3(keys: list, bucket_name: str = "bird-classification-bucket") -> list:
    """Retrieves files from S3 bucket"""
    s3 = boto3.client("s3")
    s3_files = []
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw["Body"].read()
        s3_file_image = Image.open(BytesIO(s3_file_cleaned))
        s3_files.append(s3_file_image)
    return s3_files

def load_s3_file_structure(path: str = "src/all_image_files.json") -> dict:
    """Retrieves JSON document outlining the S3 file structure"""
    with open(path, "r") as f:
        return json.load(f)

def load_list_of_images_available(all_image_files: dict, image_files_dtype: str, bird_species: str) -> list:
    species_dict = all_image_files.get(image_files_dtype)
    return species_dict.get(bird_species, [])

def predict(img, conf_rate) -> list:
    formatted_predictions = model.predict(source=[img], conf=conf_rate, save=False)
    return formatted_predictions

def image_annotation(detect_params, frame, class_list, detection_colors, slot_numbers):
    total_detections = len(detect_params[0]) if len(detect_params[0]) != 0 else 1
    if total_detections != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            class_name = class_list[int(clsID)]
            slot_number = i + 1

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            cv2.putText(
                frame,
                f"{slot_number}",
                (int(bb[0]), int(bb[1]) - 5),
                font,
                font_scale,
                (255, 255, 255),
                2,
            )
    return frame

def cal_classes_counts(total_detections: int, detect_params, class_list: dict) -> tuple:
    class_counts = {value: 0 for value in class_list.values()}
    slot_numbers = {}
    if total_detections != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            class_name = class_list[int(clsID)]
            class_counts[class_name] += 1
            slot_numbers[i + 1] = int(clsID)  # Assign slot numbers starting from 1
    return class_counts, slot_numbers

def cal_classes_percentage(total_detections: int, class_counts: dict) -> dict:
    class_percentages = {
        class_name: count / total_detections * 100
        for class_name, count in class_counts.items()
    }
    for class_name, percentage in class_percentages.items():
        print(f"Percentage of {class_name}: {percentage:.2f}%")
    return class_percentages

def save_uploaded_image(file, uploaded_path: str):
    os.makedirs(uploaded_path, exist_ok=True)
    file_path = os.path.join(uploaded_path, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

def save_image(image, image_name: str, output_path: str):
    output_directory = os.path.dirname(output_path)
    os.makedirs(output_directory, exist_ok=True)

    valid_extensions = [".jpg", ".jpeg", ".png"]
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in valid_extensions:
        output_path = os.path.splitext(output_path)[0] + image_name

    cv2.imwrite(output_path, image)

    if os.path.exists(output_path):
        print(f"Image saved successfully to: {output_path}")
    else:
        print("Failed to save the image.")




def detection_image(file):
    img = Image.open(file)
    save_uploaded_image(file, uploaded_path)

    image_path = os.path.join(uploaded_path, file.name)
    frame = cv2.imread(image_path)

    prediction = predict(frame, confidence_rate)
    class_counts, slot_numbers = cal_classes_counts(len(prediction[0]), prediction, class_list)
    predicted_image = image_annotation(prediction, frame, class_list, detection_colors, slot_numbers)
    save_image(predicted_image, file.name, predicted_path)

    class_percentage = cal_classes_percentage(len(prediction[0]), class_counts)

    file_path = os.path.join(predicted_path, file.name)
    img = Image.open(file_path)

    new_height = 550
    width, height = img.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    resized_image = img.resize((new_width, new_height))

    st.title("Detected Output")
    st.image(resized_image)
    

    # Display the dashboard
    display_dashboard(class_counts, slot_numbers)

def detection_video(video_file_path: str) -> str:
    cap = cv2.VideoCapture(video_file_path)
    frame_step = 17
    frame_position = 0
    output_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_position < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_position += frame_step
        prediction = predict(frame, confidence_rate)
        predicted_image = image_annotation(prediction, frame, class_list, detection_colors)
        output_frames.append(predicted_image)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    output_video_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        30.0,
        (output_frames[0].shape[1], output_frames[0].shape[0]),
    )

    for frame in output_frames:
        out.write(frame)

    out.release()
    return output_video_path

def save_uploaded_file(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

if __name__ == "__main__":
    uploaded_path = "uploaded_images/"
    predicted_path = "predicted_images/"

    model = load_model()
    class_list = load_index_to_label_dict()
    all_image_files = load_s3_file_structure()
    types_of_birds = sorted(list(all_image_files["test"].keys()))
    types_of_birds = [bird.title() for bird in types_of_birds]

    detection_colors = [(10, 239, 8), (252, 10, 73)]
    confidence_rate = 0.45

    st.title("Smart Parking Management!")
    instructions = """
        SPM (Smart Parking Management) aims to use machine learning, specifically the YOLO v8 algorithm, to analyze parking images and count cars and available spaces.
        """
    st.write(instructions)

    file = st.file_uploader("Upload A Video", type=["png", "jpg", "jpeg"])
    dtype_file_structure_mapping = {"Image": "image", "Video": "video"}
    data_split_names = list(dtype_file_structure_mapping.keys())

    global data_type
    data_type = "image"

    if file:
        if data_type == "image":
            detection_image(file)
        elif data_type == "video":
            file_path = save_uploaded_file(file)
            processed_video_path = detection_video(file_path)
            st.video(processed_video_path)
    else:
        data_type = st.sidebar.selectbox("Input Type", data_split_names)
        confidence_rate = st.sidebar.slider("Confidence Rate:", min_value=0, max_value=100, value=50) / 100
