# Capstone-Project-Deep-Learning-for-Computer-Vision.ipynb
# 🎓 Capstone Project: DeepFER - Facial Emotion Recognition Using Deep Learning

## 📌 Objective

The goal of this project is to build a deep learning-based system that can accurately detect human emotions from facial expressions. This system leverages Convolutional Neural Networks (CNN) and transfer learning techniques to classify facial images into seven basic emotion categories. The project demonstrates the end-to-end development of a facial emotion recognition model using real-world data.

---

## 😄 Emotion Classes

The model is trained to classify the following seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 🧠 Technologies and Tools Used

- **Language:** Python
- **Frameworks/Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn
- **Platform:** Google Colab
- **Dataset:** FER2013 (Facial Expression Recognition), in CSV format

---

## 🏗️ Project Structure

- `DeepFER.ipynb`: Main training and evaluation notebook
- `model.h5`: Trained model weights (optional)
- `images/`: Sample images for testing and result visualization
- `README.md`: Project overview and documentation
- `report.pdf`: Full report with analysis and explanation
- `demo_script.txt`: Short explanation script for presentation/demo

---

## 🧪 Model Workflow

1. **Data Loading:** FER2013 dataset loaded and preprocessed from CSV
2. **Preprocessing:** Resizing, normalization, one-hot encoding
3. **Model Building:** CNN layers with dropout and batch normalization
4. **Training:** 5–10 epochs using `model.fit()` with validation data
5. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix
6. **Testing:** Sample predictions compared to actual labels
7. **Visualization:** Performance graphs and confusion matrix plots

---

## 📊 Results

- Model achieves moderate accuracy with strong performance in identifying clearer emotions like "Happy" and "Neutral".
- Emotions like "Fear" and "Disgust" are harder to classify due to class imbalance and subtle facial features.
- Performance improves with more training time, data augmentation, and fine-tuning.

---

## 🚧 Limitations & Challenges

- Limited accuracy due to short training time and hardware limitations
- Grayscale, low-resolution (48x48) images reduce visual features
- Imbalanced dataset across emotions
- Emotion overlap (e.g., sad vs. neutral) is challenging for the model

---

## 🔮 Future Enhancements

- Train on GPU for longer durations
- Use real-time webcam integration with OpenCV
- Apply transfer learning with pre-trained models like ResNet, VGG16
- Data augmentation to balance underrepresented classes
- Deploy as a web app using Flask or Streamlit

---

## 📌 Conclusion

DeepFER successfully implements a full facial emotion recognition pipeline using deep learning. While current results are not perfect, this project forms a strong foundation for future real-time emotion-aware applications in education, healthcare, mental wellness, and human-computer interaction.

