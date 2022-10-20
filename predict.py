import os
import torch
import cv2

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