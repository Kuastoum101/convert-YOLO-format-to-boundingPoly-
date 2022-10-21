import os
import colorsys
from tqdm import tqdm
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from detector_utils import ModelConfiguration, create_detector_config, Detector
from typing import List, Tuple
#model_file_name = "centernet_recent.pt"

model_data_path = Path("../model_data/centernet_weights")#os.getcwd()
model_file_names = os.listdir(model_data_path)  # [model_file_name]*3
model_file_names = [path for path in model_file_names if path.rsplit(".")[-1] == "pt"]


model_configuration_name = "centernet_config.json"
class_labels_name = ["__background__", "person", "slip_risk", "weapon", "person1", "slip-risk", "person_on_ground",
                     "cellphone", "backpack", "broom", "suitcase", "misc", "vehicle", "reflection"]
score_threshold = 0.3
iou_threshold = 0.4

model_config = ModelConfiguration("centernet", model_configuration_name, model_file_names[0],
                                  class_labels_name)
detector_config = create_detector_config(model_config, Path(model_data_path),
                                         score_threshold=score_threshold, iou_threshold=iou_threshold)

preprocessing = detector_config.create_preprocessing_function()
# detector = Detector.create(detector_config)

class_names = detector_config.class_labels
# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75

'''model predictor functions'''
def create_detector(model_file_name):
    global model_data_path, model_configuration_name, class_labels_name, score_threshold, iou_threshold
    #model_data_path = os.getcwd()
    #_model_file_name = "../model_data/centernet_weights/" + _model_file_name
    model_config = ModelConfiguration("centernet", model_configuration_name, model_file_name,
                                       class_labels_name)
    detector_config = create_detector_config(model_config, Path(model_data_path),
                                             score_threshold=score_threshold, iou_threshold=iou_threshold)
    return Detector.create(detector_config)


detectors = None


def save_results(output_path, img_name, orig_img_rgb, text_content):
    # save result image
    save_img_path = os.path.join(output_path, img_name)
    cv2.imwrite(save_img_path, cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR))

    # save predictions
    txt_file_path = os.path.splitext(save_img_path)[0] + '.txt'
    with open(txt_file_path, 'w') as f:
        for i in text_content:
            for j in i:
                f.write(str(j))
                f.write(' ')
            f.write('\n')


def draw_predictions(orig_img_rgb, class_name, class_idx, bbox, score, font_thickness):
    global font, font_scale, colors
    x1, y1, x2, y2 = bbox
    label = f'{class_name} {score:.2f}'
    label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    if y1 - label_size[1] >= 0:
        text_origin = np.array([x1, y1 - label_size[1] - baseline])
    else:
        text_origin = np.array([x1, y1 + 1])

    for i in range(font_thickness):
        cv2.rectangle(orig_img_rgb, (x1 + i, y1 + i), (x2 - i, y2 - i), color=colors[class_idx])

    cv2.rectangle(orig_img_rgb, tuple(text_origin), tuple(text_origin + label_size + (0, baseline)),
                  color=colors[class_idx], thickness=cv2.FILLED)
    cv2.putText(
        orig_img_rgb,
        label,
        (text_origin[0], text_origin[1] + label_size[1]),
        font,
        font_scale,
        color=(0, 0, 0),
        thickness=font_thickness
    )
    return orig_img_rgb


def insert_detection_into_table(detections:tuple, table:list, frame:int, model_name:str):
    for detection in detections:
        class_idx = detection.class_id
        class_name = class_names[class_idx]
        if class_name == "__background__":
            continue
        score = detection.score
        bbox = detection.bounding_box.to_xyxy(convert_to_integers=True)
        x1, y1, x2, y2 = bbox
        table.append((frame, model_name, class_name, score, x1, y1, x2, y2))

def insert_detections_into_image(orig_img_rgb, detections):
    font_thickness = (orig_img_rgb.shape[0] + orig_img_rgb.shape[1]) // 1500
    text_content = []
    for detection in detections:
        class_idx = detection.class_id
        class_name = class_names[class_idx]
        if class_name == "__background__":
            continue
        score = detection.score
        bbox = detection.bounding_box.to_xyxy(convert_to_integers=True)
        x1, y1, x2, y2 = bbox
        text_content.append([class_name, score, x1, y1, x2, y2])
        orig_img_rgb = draw_predictions(orig_img_rgb, class_name, class_idx, bbox, score, font_thickness)

    return orig_img_rgb, text_content


def filter_detections(detections):
    for detection in detections:
        class_idx = detection.class_id
        class_name = class_names[class_idx]
        if class_name not in ["person_standing", "cellphone", "weapon"]:
            continue


def create_video_writer(vcap, out_dir, vid_name, scale_factor=0.5):
    frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out_size = (int(frame_width * scale_factor), int(frame_height * len(detectors) * scale_factor))
    FPS = int(vcap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ('M', 'J', 'P', 'G') for .avi
    out_path = os.path.join(out_dir, f'{vid_name}.mp4')
    return cv2.VideoWriter(out_path, fourcc, FPS, video_out_size), frame_width, frame_height


def create_write_csv(data,  file_name=None, fields=None):
    if file_name is None: file_name = "out.csv"
    if fields is None: fields = ["frame", "model", "label",  "score", "x1", 'y1', "x2", 'y2']
    with open(file_name, 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data)


def get_video_paths(video_dir_path):
    if os.path.isfile(video_dir_path):
        vid_names = [video_dir_path]
    else:
        vid_names = os.listdir(video_dir_path)
    vid_paths = [os.path.join(video_dir_path, x) for x in vid_names]
    return [x for x in vid_paths if os.path.isfile(x)]


def create_out_dir(dir_path: str, folder_name: str = "out_videos"):
    out_dir = (Path(dir_path).parent.resolve() / folder_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.__str__()


def predict_video(video_dir_path, bWriteVideo=False, bWriteDetection=False, bShow=False):
    global detectors, colors, preprocessing
    scale_factor = 0.5

    vid_paths = get_video_paths(video_dir_path)
    if bWriteVideo or bWriteDetection: out_dir = create_out_dir(video_dir_path)

    for vid_path in tqdm(vid_paths):
        vid_name = Path(vid_path).stem
        print(f"{vid_name} is being processed")
        vcap = cv2.VideoCapture(vid_path)

        if not vcap.isOpened(): # Check if camera opened successfully
            print("Error opening video stream or file"); continue

        out, frame_width, frame_height = create_video_writer(vcap, out_dir, vid_name, scale_factor)
        pred_acc = np.zeros((frame_height * len(detectors), frame_width, 3), dtype=np.uint8)
        # Read until video is completed
        nframe = 0
        detection_table = []
        while (vcap.isOpened()):# and nframe<10):
            # Capture frame-by-frame
            ret, orig_img = vcap.read()
            if ret == True:
                nframe += 1
                if len(orig_img.shape) > 2 and orig_img.shape[2] == 4:
                    # convert the image from RGBA2RGB
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
                orig_img_rgb = orig_img.copy()#cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                image = preprocessing(orig_img_rgb)

                hindex = 0
                for i, detector in enumerate(detectors):
                    orig_img = orig_img_rgb.copy()
                    detections = detector.detect_preprocessed(image, (1, *orig_img.shape))[0]
                    if bWriteDetection: insert_detection_into_table(detections, detection_table, nframe, model_file_names[i])
                    # filter_detections(detections)
                    if bShow or bWriteVideo:
                        orig_img, text_content = insert_detections_into_image(orig_img, detections)
                        orig_img = cv2.putText(orig_img, model_file_names[i], (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                               (0, 0, 0), 2, cv2.LINE_AA)
                        pred_acc[hindex:hindex + frame_height, :, :] = orig_img
                        hindex += frame_height
                        rescaled_img = cv2.resize(pred_acc, (0, 0), fx=scale_factor, fy=scale_factor)
                if bShow:
                    cv2.imshow('Frame', rescaled_img)  # Display the resulting frame
                    if cv2.waitKey(1) & 0xFF == ord('q'): break  # Press Q on keyboard to  exit
                if bWriteVideo: out.write(rescaled_img)

            else:
                break
        if bWriteDetection: create_write_csv(detection_table, f"{os.path.join(out_dir, vid_name)}.csv")
        # When everything done, release the video capture object
        vcap.release()
        out.release()


def predict(input_path, _detector=None, output_path=None):
    global detector, colors, preprocessing
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    if _detector is None:
        _detector = detector

    with torch.no_grad():
        if os.path.isfile(input_path):
            img_names = [input_path]
        else:
            img_names = os.listdir(input_path)
        for img_name in tqdm(img_names):
            orig_img = cv2.imread(os.path.join(input_path, img_name), cv2.IMREAD_UNCHANGED)
            if len(orig_img.shape) > 2 and orig_img.shape[2] == 4:
                # convert the image from RGBA2RGB
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
            orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            font_thickness = (orig_img.shape[0] + orig_img.shape[1]) // 1500

            image = preprocessing(orig_img)
            # We use batch_size=1 here
            detections = _detector.detect_preprocessed(image, (1, *orig_img.shape))[0]

            text_content = []

            for detection in detections:
                class_idx = detection.class_id
                class_name = class_names[class_idx]
                if class_name == "__background__":
                    continue
                score = detection.score
                bbox = detection.bounding_box.to_xyxy(convert_to_integers=True)
                x1, y1, x2, y2 = bbox
                text_content.append([class_name, score, x1, y1, x2, y2])
                orig_img_rgb = draw_predictions(orig_img_rgb, class_name, class_idx, bbox, score, font_thickness)

            if output_path is not None:
                save_results(output_path, img_name, orig_img_rgb, text_content)

    return text_content


def main(video_dir):
    global detectors, model_file_names
    detectors = [create_detector(filename) for filename in model_file_names]
    predict_video(video_dir, bWriteDetection=True, bWriteVideo=True)


if __name__ == "__main__":
    video_dir = Path("C:/Users/mpak8/Downloads/quantigo/quantigo.mp4")
    main(video_dir)
