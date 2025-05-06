# handwritten-character-recognition


This project focuses on developing a deep learning-based system for recognizing handwritten text from digital images, specifically targeting digits and letters. The project implements and compares two Convolutional Neural Network (CNN) architectures, AlexNet and LeNet-5, using the EMNIST dataset.

## Challenge Description

Handwritten text recognition involves converting images of handwritten characters or digits into machine-readable formats. This task is critical for applications such as digitizing historical documents, automating form processing, and enabling handwriting-based authentication systems. The primary challenges include:

- Variability in individual handwriting styles.
- Visually similar characters (e.g., "0" vs. "O", "1" vs. "I").
- Misalignment or noise in input images.

The objective is to train a model that accurately identifies handwritten digits (0–9) and letters from grayscale images, achieving high recognition accuracy despite these challenges.

## Requirements 

To run this project, the following dependencies are required:

- **Programming Language**: Python 3.8+
- **Libraries**:
  - TensorFlow (for model implementation and training)
  - NumPy (for numerical operations)
  - scikit-learn (for evaluation metrics)
  - Matplotlib (for plotting results and confusion matrices)
  - OpenCV (for image preprocessing and character detection)
- **Hardware**:
  - A GPU is recommended for faster training, especially for AlexNet.
  - Minimum 8 GB RAM for data processing and model training.
- **Dataset**: EMNIST dataset (details below).
- **Optional**: Kaggle API for downloading the EMNIST dataset.

Install the required libraries using pip:
```bash
pip install tensorflow numpy scikit-learn matplotlib opencv-python


## Dataset

The project utilizes the **EMNIST (Extended MNIST)** dataset, which contains grayscale images of handwritten digits (0–9) and letters (uppercase and lowercase).

- **Format**: 28x28 pixel grayscale images, single channel.
- **Classes**: 47 classes (10 digits + 37 letters, including uppercase and selected lowercase).
- **Size**: Balanced across all labels to prevent class bias.
- **Source**: Available on Kaggle at [EMNIST in CSV](https://www.kaggle.com/datasets/oderationale/mnist-in-csv).
- **Preprocessing**:
  - Corrected misaligned images by rotating vertically and flipping 90 degrees counter-clockwise (e.g., addressing "W" misinterpreted as "3").
  - Normalized pixel values to the range [0, 1].
  - For character detection, applied grayscale conversion, Gaussian blur for noise reduction, and contour detection to identify regions of interest (ROIs), resizing each to 28x28 pixels.

## Architecture

The project implements and compares two Convolutional Neural Network (CNN) models: **AlexNet** and **LeNet-5**, built using TensorFlow and trained on the preprocessed EMNIST dataset.

### AlexNet
- **Input**: 28x28x1 grayscale images.
- **Layers**:
  - 5 convolutional layers (96, 256, 384, 384, 256 filters) with ReLU activation, some followed by max-pooling (3x3, stride 2).
  - 2 fully connected layers (4096 units each) with ReLU and dropout (0.5).
  - Output layer with softmax for 47 classes.
- **Parameters**: 21,599,306 (82.39 MB).
- **Training Time**: 1 hour, 57 minutes, 29 seconds.

### LeNet-5
- **Input**: 28x28x1 grayscale images.
- **Layers**:
  - 2 convolutional layers (32, 48 filters, 5x5) with ReLU activation and max-pooling (2x2, stride 2).
  - 2 fully connected layers (256, 84 units) with ReLU activation.
  - Output layer with softmax for 47 classes.
- **Parameters**: 369,174 (1.41 MB).
- **Training Time**: 19 minutes, 3 seconds.

### Training Details
- **Method**: Train-test split with a validation set to monitor performance and prevent overfitting.
- **Optimization**: Used a callback with `save_best_only=True` to save model weights with the highest validation accuracy or lowest validation loss.
- **Data Augmentation**: Applied rotation, scaling, and flipping to enhance model robustness to handwriting variations.

## Result

The performance of AlexNet and LeNet-5 was evaluated on the EMNIST validation dataset using accuracy, precision, recall, and F1-score, computed with scikit-learn.

| Model    | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| AlexNet  | 0.98680  | 0.98687   | 0.98680 | 0.98680  |
| LeNet-5  | **0.99150** | **0.99153** | **0.99150** | **0.99149** |

### Key Findings
- **LeNet-5 Superiority**: LeNet-5 outperforms AlexNet across all metrics, likely due to its simpler architecture, which reduces overfitting and improves generalization.
- **Error Analysis**:
  - Misclassifications occur with visually similar characters (e.g., "0" vs. "O", "1" vs. "I").
  - Connected characters (e.g., two "L"s mistaken as "U") pose challenges, especially in real-world handwriting.
  - Non-dataset symbols (e.g., "!") are incorrectly matched to the closest label.
- **Character Detection**: The system accurately detects most characters but struggles with non-standard symbols and merged characters.

## Future Work

To enhance the system, the following improvements are planned:
- Re-label incorrectly predicted characters and retrain the model to boost accuracy.
- Develop word segmentation techniques for word-by-word recognition instead of character-by-character.
- Evaluate additional CNN architectures (e.g., VGG, ResNet) to compare performance against LeNet-5.
- Explore sequence-based models (e.g., RNNs with Connectionist Temporal Classification) for recognizing entire phrases or sentences.

## References

- EMNIST Dataset: [https://www.kaggle.com/datasets/oderationale/mnist-in-csv](https://www.kaggle.com/datasets/oderationale/mnist-in-csv)
- Kaggle Notebook: [https://www.kaggle.com/code/viratkothari/image-classification-of-mnist-using-vgg16/notebook](https://www.kaggle.com/code/viratkothari/image-classification-of-mnist-using-vgg16/notebook)
- OCR Tutorial: [https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow](https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow)
- Reference comparison: [https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow](https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow)

## Contributors

- Nguyễn Duy Cương (BA12-034)
- Nguyễn Ngọc Anh (BA12-015)
- Lê Tuấn Anh (BA11-005)
- Vũ Ngọc Minh (BA12-128)
- Hà Tấn Minh (BA12-126)
- Nguyễn Lân Việt (BA12-193)

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

---

For questions, issues, or contributions, please open an issue or submit a pull request on GitHub.
