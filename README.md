# E-commerce Product Image Quality Control using Deep Learning

This project automates the quality control process of product images in an e-commerce platform using deep learning. It ensures that only high-quality, relevant product images are showcased, ultimately enhancing customer experience and building trust.

---

## Features

- Image classification using a fine-tuned deep learning model  
- Filters out irrelevant or low-quality product images  
- Preprocessing pipeline to standardize and clean incoming images  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib & Seaborn (for visualization)  
- Streamlit (for deployment UI)

---

## Project Structure

ecommerce_quality_control/ ├── data/                 # Input image dataset ├── models/               # Trained models (excluded from GitHub) ├── preprocessing.py      # Image preprocessing script ├── training.py           # Model training pipeline ├── predict.py            # Image prediction logic ├── app.py                # Streamlit app for deployment ├── requirements.txt      # Python dependencies └── README.md             # Project documentation

---

## How to Run

1. *Clone the repository:*
```bash
git clone https://github.com/Harishma-K-Da/ecommerce_quality_control.git

2. Navigate to the project folder:



cd ecommerce_quality_control

3. Install required packages:



pip install -r requirements.txt

4. (Optional) Run preprocessing and training scripts:



python preprocessing.py
python training.py

5. Run the Streamlit web app:



streamlit run app.py


---

Note

Make sure your dataset is inside the data/ folder.

Trained model files are not uploaded to GitHub — you should place them manually inside the models/ folder.



---

## Download the Model

The fine-tuned model file is large, so it's hosted on Google Drive.  
[Click here to download product_damage_classifier.h5](https://drive.google.com/uc?export=download&id=10ZzSQ_Pn_2Xqai8ZYs3p_-_YRqmO6JlB)


---

License

This project is for educational and demonstration purposes only.