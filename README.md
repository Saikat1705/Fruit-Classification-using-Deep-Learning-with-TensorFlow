
# 🍎🍌 Fruit Classification using Deep Learning

This project classifies fruit images using a Convolutional Neural Network (CNN) model built with TensorFlow/Keras. It identifies fruits like apples, bananas, oranges, and more from image input.

---

## 📂 Dataset

The dataset consists of images of various fruits categorized into labeled folders:

```

dataset/
│
├── train/
│   ├── Apple/
│   ├── Banana/
│   ├── Orange/
│   └── ... (other fruit classes)
│
└── test/
├── Apple/
├── Banana/
├── Orange/
└── ...

````

Each folder contains images corresponding to that fruit class.

---

## 🧠 Model Overview

We use:
- **Convolutional Neural Networks (CNNs)** for image feature extraction
- **ImageDataGenerator** for data augmentation
- **Softmax activation** for multi-class classification

---

## 🚀 How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fruit-classification.git
    ````

2. Navigate to the project directory:

   ```bash
   cd fruit-classification
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook:

   ```bash
   jupyter notebook
   ```

---

## 🧪 Prediction Example

You can test the model on a single image after training:

  ```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("test_image.jpg", target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
print(f"Predicted Fruit: {predicted_class}")
   ```

---

## 📈 Accuracy

The model typically achieves **high accuracy** on test data with proper tuning and augmentation.

---

## 🛠️ Requirements

  ```bash
pip install tensorflow numpy matplotlib seaborn
  ```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* Dataset Source: [Kaggle Fruit Classification Dataset](https://www.kaggle.com/datasets)
* Deep learning powered by TensorFlow and Keras

