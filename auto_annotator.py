import os
import cv2
import predict
from pascal_voc_writer import Writer
from pathlib import Path
import shutil
import xmltodict
import intersect_bboxes


class_dict = {'person_fallen': 0, 'person_sitting': 1, 'person_standing': 2, 'snow': 3, 'spill': 4, 'vehicle': 5,
              'weapon': 6, 'unknown': 7}

abbr_dict = {'person_fallen': 'PFN', 'person_sitting': 'PSI', 'person_standing': 'PST', 'snow': 'SN', 'spill': 'SP',
             'vehicle': 'VHC', 'weapon': 'WPN', 'unknown': 'UKN'}

cls2centernet_dict = {'person_fallen': 'person_on_ground', 'person_sitting': 'person1', 'person_standing': 'person',
                      'snow': 'slip-risk', 'spill': 'slip_risk', 'vehicle': 'vehicle', 'weapon': 'weapon',
                      'unknown': 'unknown', 'handgun': 'weapon'}

class_id2str_dict = {v: k for k, v in class_dict.items()}
centernet2cls_dict = {v: k for k, v in cls2centernet_dict.items()}
centernet2int_dict = {k: i for k, i in zip(cls2centernet_dict.values(), range(len(cls2centernet_dict)))}
waitkey_ms = 2000


def execute_center_detection(path):
    center_preds = predict(path)
    return centernet_parser(center_preds)


def contains(rect, point):
    return ((rect[0] <= point[0]) & (rect[2] >= point[0])) and ((rect[1] <= point[1]) & (rect[3] >= point[1]))


def convert_coord(coords, img):
    h, w = img.shape[:2]
    if not isinstance(coords, list):
        coords = [coords]
    for i, coord in enumerate(coords):
        x, y = coord
        x = int(float(x) * (float(w) / wnd_size[0]) + 0.5)
        y = int(float(y) * (float(h) / wnd_size[1]) + 0.5)
        coords[i] = [x,y]
    if len(coords)>1:
        if coords[0][0] > coords[1][0]:
            coords[0][0], coords[1][0] = coords[1][0], coords[0][0]
        if coords[0][1] > coords[1][1]:
            coords[0][1], coords[1][1] = coords[1][1], coords[0][1]
    return coords


def select_rectangles_todelete(inp_img, inp_bboxes):
    global rightclck, cv_wnd_name, wnd_size
    rightclck = None
    while rightclck is None:
        print("please right click on one of existing rectangles on images to delete")
        cv2.waitKey(1000)
    rightclck = convert_coord(rightclck, inp_img)[0]
    index = None
    rcolor = (0, 0, 255)
    bcolor = (255, 0, 0)
    org_img = inp_img.copy()
    for i, bbox in enumerate(inp_bboxes):
        x1, y1, x2, y2 = bbox
        if contains(bbox, rightclck):
            cv2.rectangle(inp_img, (x1, y1), (x2, y2), bcolor, 2)
            index = i
        else:
            cv2.rectangle(inp_img, (x1, y1), (x2, y2), rcolor, 2)
    cv2.imshow(cv_wnd_name, cv2.resize(inp_img, wnd_size))
    print("please hit 'o' key to verify or 'r' key to retry or any key to continue")
    key = cv2.waitKey(-1)
    return index, key if key != ord('r') else select_rectangles_todelete(org_img, inp_bboxes)

def select_rectangles_toupdate(inp_img, inp_bboxes):
    global leftclck, cv_wnd_name, wnd_size
    leftclck = []
    while len(leftclck)<2:
        print("please left clicks two times on one of existing rectangles or new rectangles on images to update/add new")
        cv2.waitKey(1000)
    leftclck = convert_coord(leftclck, inp_img)
    sel_bbox =[leftclck[0][0], leftclck[0][1], leftclck[1][0], leftclck[1][1]]
    index = None
    rcolor = (0, 0, 255); bcolor = (255, 0, 0)
    org_img = inp_img.copy()
    for i, bbox in enumerate(inp_bboxes):
        x1, y1, x2, y2 = bbox
        if intersect_bboxes(bbox, sel_bbox):
            index = i
        cv2.rectangle(inp_img, (x1, y1), (x2, y2), rcolor, 2)
    cv2.rectangle(inp_img, (sel_bbox[:2]), (sel_bbox[2:]), bcolor, 2)
    cv2.imshow(cv_wnd_name, cv2.resize(inp_img, wnd_size))
    print("please hit 'o' key to verify or 'r' key to retry or any key to continue")
    key = cv2.waitKey(-1)
    return index, key if key != ord('r') else select_rectangles_todelete(org_img, inp_bboxes)

def delete_rectangle(img, bboxes, preds):
    index, key = select_rectangles_todelete(img, bboxes)
    if index is not None and key == ord('o'):
        del bboxes[index]
        del preds[index]
    else:
        print("continues to verification")
    return bboxes, preds


def update_rectangle(img, bboxes, preds):
    global leftclck
    index, key = select_rectangles_toupdate(img, bboxes)
    sel_box = [leftclck[0][0], leftclck[0][1], leftclck[1][0], leftclck[1][1]]
    if index is not None and key == ord('o'):
        bboxes[index] = sel_box
    elif index is None and key == ord('o'):
        bboxes.append(sel_box)
        preds.append(preds[0])
    else:
        print("continues to verification")
    return bboxes, preds


def update_pred_boxes(key, _image, _preds, _bboxes):
    org_img = _image.copy()
    global cv_wnd_name, waitkey_ms, rightclck, leftclck
    if key == 32:
        cv2.imshow(cv_wnd_name, cv2.resize(_image, wnd_size))
        print("press 'u' to update and 'd' to delete")
        key1 = cv2.waitKey(-1)
        if key1 == ord('u'):
            print("please 2 times left click to create a new rectangle on existing rectangles or "
                  "missing target object")
            update_rectangle(org_img, _bboxes, _preds)
        elif key1 == ord('d'):
            print("please right click on existing rectangle to delete")
            delete_rectangle(org_img, _bboxes, _preds)

    return _bboxes, _preds

counter = 0
def draw_preds(_image, _preds, _scores, _bboxes):
    global waitkey_ms
    color = (0, 0, 255)

    for pred, score, bbox in zip(_preds, _scores, _bboxes):
        if pred != "weapon":
            continue
        x1, y1, x2, y2 = bbox
        texts = [" pred :" + pred,  " score: " + str(round(score, 2))]
        pix_step = 20
        y_cord = y2 + pix_step
        for text in texts:
            cv2.putText(img=_image, text=text, org=(x1, y_cord), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6,
                        color=color, thickness=1)
            y_cord += pix_step
        cv2.rectangle(_image, (x1, y1), (x2, y2), color, 2)
        cv2.imshow(cv_wnd_name, cv2.resize(_image, wnd_size))
        #image_writer(_image)
    return cv2.waitKey(waitkey_ms)

def image_writer(_image):
    global counter
    counter += 1
    filename = "image{:03d}.jpeg".format(counter)
    cv2.imwrite(filename, _image)

def centernet_parser(inp_center_preds):
    preds = []; scores = []; bboxes = []
    for center_pr in inp_center_preds:
        pred, score, *bbox, = center_pr
        preds.append(pred), scores.append(score), bboxes.append(tuple(bbox))

    return preds, scores, bboxes


def accumulator(tr_labels, clf_labels, cent_labels):
    global y_true, y_center, y_clfs, centernet2int_dict
    for lab, y_c, y_cen in zip(tr_labels, clf_labels, cent_labels):
        y_true.append(centernet2int_dict[lab])
        y_clfs.append(centernet2int_dict[y_c])
        y_center.append(centernet2int_dict[y_cen])

'''move img into not annotated when xml is deleted'''
def update_xml_file(img_path, img_name, _preds, _bboxes):
    global not_annot_dir_path
    writer = None
    xml_path = os.path.join(xml_dir_path, img_name.split(".")[0]) + ".xml"
    if len(_preds) < 1:
        os.remove(xml_path)
        shutil.move(img_path, not_annot_dir_path)
        return writer
    return create_xml_file(img_path, img_name, _preds, _bboxes)


def create_xml_file(img_path, img_name, _preds, _bboxes):
    global xml_dir_path
    writer = None
    xml_path = os.path.join(xml_dir_path, img_name.split(".")[0]) + ".xml"
    for pred, bbox in zip(_preds, _bboxes):
        if pred == "weapon":
            if writer is None:
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                writer = Writer(img_path, w, h)
            writer.addObject(pred, *bbox)
    if writer is not None:
        writer.save(xml_path)
    return writer


def xml_annotation_generator(_img_dir_path):
    global xml_dir_path
    _img_dir_path = Path(_img_dir_path)

    parent_path = _img_dir_path.parent.absolute()
    xml_dir_path = os.path.join(parent_path, "xml")
    not_annot_dir_path = os.path.join(parent_path, "not_annotated")
    Path(xml_dir_path).mkdir(parents=True, exist_ok=True)
    Path(not_annot_dir_path).mkdir(parents=True, exist_ok=True)
    img_paths = os.listdir(_img_dir_path)
    for img_name in img_paths:
        img_path = os.path.join(_img_dir_path, img_name)
        preds, scores, bboxes = execute_center_detection(img_path)
        if len(preds) < 1:
            shutil.move(img_path, not_annot_dir_path)
            continue
        writer = create_xml_file(img_path, img_name, preds, bboxes)

        if writer is None:
            shutil.move(img_path, not_annot_dir_path)
        else:
            img = cv2.imread(img_path)
            draw_preds(img, preds, scores, bboxes)


    return not_annot_dir_path, xml_dir_path


cv_wnd_name = "images"
wnd_size = (960, 540)
cv2.namedWindow(cv_wnd_name, cv2.WINDOW_NORMAL)
rightclck = None
leftclck = []


def mouse_click(event, x, y, flags, param):
    global leftclck, rightclck
    if event == cv2.EVENT_RBUTTONDOWN:
        rightclck = [x, y]
    elif event == cv2.EVENT_LBUTTONDOWN:
        if len(leftclck)>1:
            leftclck = []
        leftclck.append([x, y])


cv2.setMouseCallback(cv_wnd_name, mouse_click)
img_dir_path = 'C:/Users/mpak8/Downloads/pyramid' #"D:/intellisee_data/model_eval/broadlawn_test/quantigo_guns_cleaned" #"D:\intellisee_data\guns_google\images"  # sys.argv[1]
bgenerate_annotations = False
if bgenerate_annotations:
    not_annot_dir_path, xml_dir_path = xml_annotation_generator(img_dir_path)
else:
    '''verification is started here'''
    xml_dir_path = 'D:/intellisee_data/model_eval/broadlawn_test/xml'#'C:/Users/mpak8/Downloads/xml' #"D:/intellisee_data/model_eval/broadlawn_test/xml"
    not_annot_dir_path = 'D:/intellisee_data/model_eval/broadlawn_test/not_annotated'

xml_dir_path = Path(xml_dir_path)
not_annot_dir_path = Path(not_annot_dir_path)
cv2.setMouseCallback(cv_wnd_name, mouse_click)


xml_file_list = os.listdir(xml_dir_path)
for xml_path in xml_file_list:
    xml_path = os.path.join(xml_dir_path, xml_path)
    with open(xml_path) as file:
        file_data = file.read()  # read file contents

    # parse data using package
    dict_data = xmltodict.parse(file_data)
    filepath = dict_data['annotation']['path']
    filename = dict_data['annotation']['filename']
    img = cv2.imread(filepath)
    video_size = dict_data['annotation']['size']
    detect_objs = dict_data['annotation']['object']
    if not isinstance(detect_objs, list):
        detect_objs = [detect_objs]

    bboxes = []; scores = []; preds = []
    for obj in detect_objs:
        bbox = [int(cord) for cord in obj['bndbox'].values()]
        class_name = obj['name']
        bboxes.append(bbox)
        preds.append(class_name)
        scores.append(-1.0)
    key = draw_preds(img, preds, scores, bboxes)
    bboxes, preds = update_pred_boxes(key, img, preds, bboxes)
    update_xml_file(filepath,filename, preds, bboxes)
