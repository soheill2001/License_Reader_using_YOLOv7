from tensorflow.keras.models import load_model
from models.experimental import attempt_load
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
import cv2
import numpy as np
import torch
import argparse
from rotate_image import find_longest_line, find_line_angle, rotate_image, adjust_cropping
from ocr import Build_Model, Dataset, Confusion_Matrix

def Load_Model(yolo_weigth, ocr_weight):
    OCR = load_model(ocr_weight)
    yolo = attempt_load(yolo_weigth)
    stride = int(yolo.stride.max())
    device = select_device('cpu')
    half = device.type != 'cpu'
    yolo = TracedModel(yolo, device, 640)
    if half:
        yolo.half()
    yolo.eval()
    return OCR, yolo, stride, device, half

def Prediction(image_path, stride, device, half):
    source_image = cv2.imread(image_path)
    img = letterbox(source_image, 640, stride=stride)[0]
    img = img[:, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = yolo(img, augment=True)[0]
    pred = non_max_suppression(pred, classes=0, agnostic=True)
    return pred, img, source_image

def Detect_Plates(pred, img, source_image):
    plate_detections = []
    det_confidences = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())
    return plate_detections, det_confidences

def Rotate_Image(plate_image_gr):
    linessorted = find_longest_line(plate_image_gr)
    rot_angle = find_line_angle(linessorted[-1])
    rotated_img = rotate_image(plate_image_gr, rot_angle)
    cropped_rotated_img = adjust_cropping(rotated_img)
    cw = cropped_rotated_img.shape[1]
    return cropped_rotated_img

def Extract_Plate_Letters(cropped_rotated_img):
    h, w = cropped_rotated_img.shape
    chopfactors = [(80, 160), (150, 220), (210, 310), (310, 380), (380, 430), (430, 500), (500, 560), (540, 610)]
    plate_letters= []
    for factor in chopfactors:
        w1 = int((factor[0] / 600)*w)
        w2 = int((factor[1] / 600)*w)
        letterpatch = cropped_rotated_img[:,w1:w2]
        letterpatch_resized = cv2.resize(letterpatch, (28, 28), interpolation= cv2.INTER_LINEAR)
        plate_letters.append(letterpatch_resized)
    plate_letters = np.array(plate_letters)
    return plate_letters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weight_path', type=str)
    parser.add_argument('--ocr_weight_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--ocr_train_path', type=str, default="")
    parser.add_argument('--ocr_test_path', type=str, default="")
    args = parser.parse_args()
    
    if args.ocr_train_path != "" and args.ocr_test_path != "":
        OCR = Build_Model()
        train_images, train_labels, test_images, test_labels = Dataset(args.ocr_train_path, args.ocr_test_path)
        OCR.fit(train_images, train_labels, epochs=200, validation_split=0.2)
        Confusion_Matrix(OCR, test_images, test_labels)
    else:
        OCR, yolo, stride, device, half = Load_Model(args.yolo_weight_path, args.ocr_weight_path)
        pred, img, source_image = Prediction(args.image_path, stride, device, half)
        plate_detections, det_confidences = Detect_Plates(pred, img, source_image)

        index = np.argmax(np.array(det_confidences))
        coord = plate_detections[0]
        cropped_image = source_image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
        cv2.imwrite("cropped_image.jpg", cropped_image)
        plate_image_gr = cv2.imread("cropped_image.jpg", 0)
        plate_image_gr = np.array(plate_image_gr)

        cropped_rotated_img = Rotate_Image(plate_image_gr)
        plate_letters = Extract_Plate_Letters(cropped_rotated_img)

    number_plate = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'G', 'H', 'J', 'L', 'M', 'N', 'S', 'T', 'V', 'Y']
    predictions = OCR.predict(plate_letters)
    plate = [number_plate[k] for k in list(np.argmax(predictions, axis=1))]
    print(plate)