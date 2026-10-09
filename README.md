# Traffic Sign Classification System using C++ and ONNX Runtime

A Machine Learning model deployment project that classifies traffic sign images using a CNN model trained in Python and deployed in a C++ application using ONNX Runtime and OpenCV.

## Project Overview

This project demonstrates the deployment of a trained Machine Learning image classification model into a C++ application.

A Convolutional Neural Network (CNN) is trained using the German Traffic Sign Recognition Benchmark (GTSRB) dataset in Python. The trained PyTorch model is exported to ONNX format and deployed using C++ with ONNX Runtime and OpenCV.

The C++ application takes a traffic sign image as input and provides:

- Predicted traffic sign class
- Prediction confidence
- Inference time

## Problem Statement

Development of an image classification system that deploys a trained machine learning model in a C++ application to classify input images and provide prediction results with confidence and inference time.

## Objectives

- Train a CNN model for traffic sign classification.
- Use the GTSRB dataset for training and testing.
- Export the trained PyTorch model to ONNX format.
- Deploy the ONNX model in a C++ application.
- Perform image preprocessing using OpenCV.
- Perform inference using ONNX Runtime.
- Calculate prediction confidence.
- Measure model inference time.

## System Architecture

```text
GTSRB Dataset
      |
      v
Python / PyTorch CNN Training
      |
      v
Trained PyTorch Model
      |
      v
ONNX Export
      |
      v
traffic_sign_cnn.onnx
      |
      v
C++ Application
      |
      v
OpenCV Preprocessing
      |
      v
ONNX Runtime
      |
      v
Prediction + Confidence + Inference Time
Project Workflow
Input Traffic Sign Image
          |
          v
     Load Image
          |
          v
    Resize to 32x32
          |
          v
      BGR to RGB
          |
          v
   Pixel Normalization
          |
          v
       HWC to CHW
          |
          v
    Create Tensor
          |
          v
   ONNX Runtime
     Inference
          |
          v
    Model Output
          |
          v
  Predicted Class
          |
          v
 Confidence + Time
Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

Dataset details:

Number of classes: 43
Task: Multi-class traffic sign classification
Input size: 32 x 32 pixels
Machine Learning Model

A Convolutional Neural Network (CNN) was developed using PyTorch.

CNN Architecture
Input Image
32 x 32 x 3
     |
     v
Convolution Layer
32 Filters
     |
     v
ReLU
     |
     v
Max Pooling
     |
     v
Convolution Layer
64 Filters
     |
     v
ReLU
     |
     v
Max Pooling
     |
     v
Flatten
     |
     v
Fully Connected Layer
128 Neurons
     |
     v
Output Layer
43 Classes
Training Configuration
Parameter	Value
Framework	PyTorch
Dataset	GTSRB
Input Size	32 x 32
Number of Classes	43
Batch Size	32
Optimizer	Adam
Learning Rate	0.001
Loss Function	Cross Entropy Loss
Epochs	5
Model Performance
Metric	Result
Training Accuracy	98.32%
Test Accuracy	85.17%
Final Training Loss	0.0598

The difference between training and test accuracy indicates some overfitting. This can be improved using data augmentation, regularization and a more advanced CNN architecture.

Model Deployment

The trained PyTorch model is exported to ONNX format.

PyTorch Model
     |
     v
traffic_sign_cnn.pth
     |
     v
ONNX Export
     |
     v
traffic_sign_cnn.onnx
     |
     v
C++ Application
C++ Deployment

The C++ application uses:

OpenCV

OpenCV is used for:

Loading the image
Resizing the image
Converting BGR to RGB
Pixel preprocessing
ONNX Runtime

ONNX Runtime is used for:

Loading the ONNX model
Creating input tensors
Running inference
Obtaining model output
Image Preprocessing

The input image goes through these steps:

Load the image.
Resize it to 32 x 32 pixels.
Convert BGR to RGB.
Normalize pixel values from 0-255 to 0-1.
Convert image format from HWC to CHW.
Create the input tensor.

The final input tensor shape is:

[1, 3, 32, 32]
Prediction and Confidence

The model produces output values for all 43 classes.

The class with the highest output value is selected as the predicted class.

Softmax is then used to calculate the prediction confidence.

Inference Time

The C++ application measures the time required by ONNX Runtime to perform model inference.

This helps evaluate the speed of the deployed Machine Learning model.

Sample Output
Image loaded successfully!
ONNX model loaded successfully!

==============================
Traffic Sign Classification
==============================
Predicted Class : 12
Confidence      : 100%
Inference Time  : 3.8592 ms
==============================

## Project Demo

![Traffic Sign Classification C++ Demo](screenshots/project_demo.png)
Project Structure
ML-Model-Deployment-CPP/
|
├── cpp/
│   ├── main.cpp
│   ├── test_opencv.cpp
│   ├── test_onnx.cpp
│   ├── traffic_sign_classifier.exe
│   └── onnxruntime.dll
|
├── python/
│   ├── train.py
│   ├── export_model.py
│   ├── test_onnx.py
│   └── requirements.txt
|
├── models/
│   ├── traffic_sign_cnn.pth
│   └── traffic_sign_cnn.onnx
|
├── data/
├── tests/
├── benchmarks/
├── docs/
├── screenshots/
├── test_image.jpg
├── README.md
└── .gitignore
Technologies Used
C++
Python
PyTorch
Torchvision
Machine Learning
Convolutional Neural Networks
ONNX
ONNX Runtime
OpenCV
CMake
Git
GitHub
Installation Requirements
Python 3.x
C++ compiler
OpenCV
ONNX Runtime
CMake
Git
PyTorch
Torchvision
ONNX
ONNX Runtime Python package
```text
```
## How to Run

### Prerequisites

Before running the project, install and configure:

* Python 3.11 or a compatible Python version
* Git
* MSYS2 UCRT64 with GCC/G++
* OpenCV
* ONNX Runtime C++ library

The C++ compilation commands below assume that OpenCV and ONNX Runtime are installed in the MSYS2 UCRT64 environment.

### 1. Clone the Repository

```bash
git clone https://github.com/SiddhiZire/ML-Model-Deployment-CPP.git
cd ML-Model-Deployment-CPP
```

### 2. Set Up the Python Environment

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r python/requirements.txt
pip install onnxscript
```

### 3. Train the Model

```powershell
python python/train.py
```

This downloads the GTSRB dataset if necessary, trains the CNN model, evaluates its test accuracy, and saves the trained model to:

`models/traffic_sign_cnn.pth`

Ensure the `models` directory exists before saving the model.

### 4. Export the Model to ONNX

```powershell
python python/export_model.py
```

This exports the trained model to:

`models/traffic_sign_cnn.onnx`

### 5. Test the ONNX Model

```powershell
python python/test_onnx.py
```

This checks whether the ONNX model loads and performs inference successfully.

### 6. Add a Test Image

Place a traffic-sign image in the project root and name it:

`test_image.jpg`

The image must be readable by OpenCV.

### 7. Compile the C++ Classifier

Open the MSYS2 UCRT64 terminal in the project root and run:

```bash
g++ cpp/main.cpp -o cpp/traffic_sign_classifier.exe \
-I/ucrt64/include/opencv4 \
-I/ucrt64/include/onnxruntime \
-L/ucrt64/lib \
-lonnxruntime \
$(pkg-config --cflags --libs opencv4)
```

Make sure the required ONNX Runtime DLL is available at runtime. The executable may need the correct DLL in its runtime search path.

### 8. Run the C++ Classifier

From the project root, run:

```bash
./cpp/traffic_sign_classifier.exe
```

The program displays the predicted class, traffic-sign name, confidence score, and inference time.

## Testing

The project includes tests for:

OpenCV C++ installation
ONNX Runtime C++ installation
ONNX model execution

Test files:

cpp/test_opencv.cpp
cpp/test_onnx.cpp
python/test_onnx.py
Key Learning Outcomes

This project demonstrates:

Machine Learning model training
CNN-based image classification
Computer Vision
Model export
ONNX model conversion
C++ Machine Learning deployment
OpenCV image preprocessing
ONNX Runtime inference
Tensor manipulation
Softmax probability calculation
Inference performance measurement
Git and GitHub
Future Improvements
Improve test accuracy
Reduce overfitting
Add data augmentation
Use a deeper CNN architecture
Display actual traffic sign names
Add real-time camera input
Add a graphical user interface
Add model quantization
Optimize inference speed
Add automated testing
Real-World Applications

Traffic sign classification can be used as a component of:

Autonomous driving systems
Advanced Driver Assistance Systems (ADAS)
Intelligent transportation systems
Road safety applications
Smart vehicle systems
Traffic monitoring systems

## Author

Siddhi Zire

B.Tech Computer Science and Engineering – Artificial Intelligence

## License

This project is created for educational, learning and portfolio purposes.
