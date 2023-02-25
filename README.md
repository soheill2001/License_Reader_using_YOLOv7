# License Plate Recognition using YOLOv7

This code implements a license plate recognition system using a combination of object detection and optical character recognition (OCR) techniques.

The system takes an input image of a car and first uses a pre-trained YOLOv7 object detection model to detect the license plate. Then, it performs image processing and rotation on the detected plate to obtain an aligned version of the plate. Finally, the aligned plate is passed to a pre-trained OCR model to extract the characters from the plate.

![Car Image](https://drive.google.com/uc?export=view&id=<car image id>)
![License Plate Image](https://drive.google.com/uc?export=view&id=<license plate image id>)

## Prerequisites
The following Python packages are required to run this code:
```
TensorFlow
OpenCV
Numpy
PyTorch4
Keras
Matplotlib
Seaborn
math
```

## Usage
The code has two modes of operation, training and prediction.

### Training
To train the Optical Character Recognition (OCR) model, use the following command:

```
python persian_plates.py --ocr_train_path <path to train dataset> --ocr_test_path <path to test dataset>
```
Where <path to train dataset> and <path to test dataset> are the paths to the train and test datasets respectively.

### Prediction
To predict the license plate in an input image, use the following command:

```
python persian_plates.py --yolo_weight_path <path to YOLOv7 weight file> --ocr_weight_path <path to OCR weight file> --image_path <path to input image>
```
Where <path to YOLOv7 weight file> is the path to the YOLOv7 weight file, <path to OCR weight file> is the path to the OCR weight file, and <path to input image> is the path to the input image.

Pre-trained weights are in `weight` folder.

## Challenges
During the training of the YOLOv7 model, one of the challenges was finding a dataset for license plates that is similar to Persian plates. To overcome this challenge, foreign plates that are similar to Persian plates were used, and a dataset of Persian plates was found.

Another challenge during training the OCR model was finding a dataset for each Persian number. A dataset containing 30,000 images, including numbers and alphabets for Persian plates, was finally found.

However, after detecting the license plate bounding boxes using YOLO, the OCR model may not predict the numbers or alphabets correctly. This is because the dataset is not enough, and the quality of the images for Persian plates is not good enough for the OCR model to understand.

## Document Files

### OCR
This code implements a CNN for image classification using TensorFlow and Keras libraries. The CNN architecture consists of two convolutional layers with ReLU activation function, followed by a max pooling layer, a flatten layer, and two fully connected layers with ReLU and softmax activation functions.


The code includes three main functions:

+ `Build_Model()`: defines and compiles the CNN model
+ `Dataset`: loads and preprocesses the dataset from two directories containing images and their labels
+ `Confusion_Matrix`: predicts labels of test images using the trained model and visualizes the confusion matrix.


The dataset should be organized in two directories: one for training images and labels, and another for testing images and labels. Each image should have a corresponding label file with the same name, containing a single integer representing the class label. Example:

```
train/
├── images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── ...

test/
├── images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── ...
```

### Rotate Images
The Longest Line Detector and Image Rotator is a Python implementation of a computer vision algorithm to detect the longest line in an image and rotate the image accordingly. This algorithm can be useful in various applications, such as detecting the orientation of documents, license plates, or other objects with prominent lines.

#### Functionality
The main functionality of the code is contained within the `find_longest_line` and `rotate_image` functions. The `find_longest_line` function takes an image as input and returns the longest line detected in the image, while the `rotate_image` function takes an image and an angle as inputs and returns the rotated image.

Additionally, the `adjust_cropping` function can be used to crop the rotated image to a desired size.


### Persian Plates
#### Workflow
The code has the following workflow:

+ Load the YOLOv7 object detection model and the OCR model.
+ Load the input image and resize it to a fixed size.
+ Use YOLOv7 to detect license plates in the image.
+ Select the license plate with the highest confidence score.
+ Crop the license plate from the image.
+ Rotate the license plate so that it is horizontally aligned.
+ Segment the license plate into individual characters.
+ Use the OCR model to recognize each character.
+ Combine the recognized characters to form the license plate number.


## Conclusion
This code implements a license plate recognition system using a combination of object detection and OCR techniques. It takes an input image of a car, detects the license plate, aligns it properly and extracts the characters from the plate. The code can be further optimized by fine-tuning the YOLOv7 and OCR models on specific datasets for better accuracy.
