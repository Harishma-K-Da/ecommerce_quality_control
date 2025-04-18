# E-commerce Product Image Quality Control using Deep Learning

This project automates the quality control process of product images in an e-commerce platform using deep learning. It ensures that only high-quality, relevant product images are showcased, ultimately enhancing customer experience and building trust.



## Features

- Image classification using a fine-tuned deep learning model
- Filters out irrelevant or low-quality product images
- Preprocessing pipeline to standardize and clean incoming images


## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy & Pandas
- Matplotlib & Seaborn (for visualization)



## Project Structure

ecommerce_quality_control/ │ ├── data/                  # Input image dataset ├── models/                # Trained models (excluded from GitHub) ├── preprocessing.py       # Image preprocessing script ├── training.py            # Model training pipeline ├── predict.py             # Image prediction logic ├── requirements.txt       # Python dependencies └── README.md              # Project documentation (this file)


## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Harishma-K-Da/ecommerce_quality_control.git

2. Navigate to the project folder:



cd ecommerce_quality_control

3. Install the required packages:



pip install -r requirements.txt

4. Run preprocessing and training scripts as needed:



python preprocessing.py
python training.py


Note

Make sure the dataset is placed inside the data/ folder.

The trained model files are not uploaded to GitHub (add them to models/ locally).


License

This project is for educational and demonstration purposes only.




