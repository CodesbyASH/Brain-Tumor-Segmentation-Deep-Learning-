# Brain-Tumor-Segmentation-Deep-Learning-


📌Overview

This project utilizes **deep learning and image processing techniques** to classify and segment brain tumors from MRI scans. Built using the **Brain Tumor MRI Dataset from Kaggle**, it detects four tumor classes: **glioma, meningioma, pituitary, and no tumor**. The pipeline includes **image preprocessing, CNN-based classification**, and **K-means-based segmentation** to visually localize tumor areas.



🎯Objectives

- Classify MRI images into one of four categories.
- Segment the tumor region within the scan.
- Apply robust preprocessing techniques to improve model performance.
- Evaluate model performance using metrics like accuracy, loss, and F1-score.

---

Dataset

- Source: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Classes: `glioma`, `meningioma`, `pituitary`, `no tumor`
- Format: MRI scans organized in labeled folders for training and testing.

---

Tools and Libraries

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV, Scikit-learn
- Matplotlib / Seaborn

  ![image](https://github.com/user-attachments/assets/69e5896d-c56d-4692-8cae-f5e8f23f7ada)




---

Image Processing Pipeline

 1. Preprocessing
- **Resizing** all images to `128*128`
- **Normalization** to scale pixel values between 0–1
- **Windowing** for enhancing soft tissue contrast
- **Histogram Analysis** for intensity distribution
- **Metadata Handling** using DataFrames for analysis (voxel size, scan quality)

  ![image](https://github.com/user-attachments/assets/bcc2ce39-442a-4299-903c-f1954ebe3b83)


  ![image](https://github.com/user-attachments/assets/1fcb6875-7d59-499f-a63d-90605d886242)




 
 2. **Data Augmentation**
- Rotation
- Horizontal/Vertical flipping
- Brightness adjustments
- Zooming/scaling



 Model Architecture (CNN)

```plaintext
Input Layer (128x128x3)
→ Conv2D + ReLU
→ MaxPooling
→ Conv2D + ReLU
→ MaxPooling
→ Flatten
→ Dense Layer + ReLU
→ Dropout
→ Dense Layer (Softmax Output - 4 Classes)

![image](https://github.com/user-attachments/assets/dc7edfe5-3496-4e8f-997a-a1a75b098424)





Loss Function: Categorical Crossentropy

Optimizer: Adam

Activation: ReLU + Softmax

Epochs: 20


 Segmentation Approach
K-means Clustering (k=2): Segment MRI images into tumor vs non-tumor regions based on intensity.

Morphological Operations: Erosion, dilation to refine segmentation.

Final output: Overlay mask on original scan.


📊 Results
Classification Metrics:

Class	Precision	Recall	F1-score
Glioma	0.95	0.88	0.91
Meningioma	0.88	0.78	0.82
Pituitary	0.95	1.00	0.95
No Tumor	0.93	0.99	0.96

![image](https://github.com/user-attachments/assets/83c7762d-6429-45f1-87ee-79979088800e)



Sample Visuals
Classification confusion matrix

Accuracy/loss plots

![image](https://github.com/user-attachments/assets/16b7904a-3504-4e74-b6fa-99921bd4caa8)


Original MRI + segmented tumor region (K-means output)

![image](https://github.com/user-attachments/assets/40bf3cc2-ead2-4a90-a9c8-69422c7b588f)


![image](https://github.com/user-attachments/assets/c30de938-86d8-468b-a865-91f47a3a8ec7)



💡 Lessons Learned
Importance of preprocessing steps (resizing, normalization)

K-means as a starting segmentation method, with future potential for U-Net.

Class imbalance impacts metrics like recall, especially in meningioma cases.

Deployment of CNN model and its application.


🚀 Future Work
Integrate U-Net for enhanced pixel-level segmentation.

Use 3D volumetric MRI data for better depth representation.

Implement real-time Flask/Streamlit deployment.

Add metadata-aware classification for patient-level insights.


🤝 Acknowledgements
Dataset: Sartaj Bhuvaji via Kaggle

Guided by course: Emerging Trends in Data Technology(TAFE NSW)

Special thanks to the deep learning community and all the open source contributors.
