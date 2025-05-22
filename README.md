
# 🧰 Digital Image Processing Toolbox

A web-based interactive toolbox for performing a variety of digital image processing operations, built using **Flask**, **OpenCV**, and **NumPy**. This project provides a clean and intuitive GUI where users can upload an image and apply several transformations and analysis techniques, grouped by category.

![UI Preview](./path-to-your-screenshot.png)

## 🚀 Features

The application supports a wide range of image processing operations:

### 🔹 1. Point Operations
- Addition
- Subtraction
- Division
- Complement

### 🎨 2. Color Operations
- Change red channel intensity
- Swap color channels (e.g., Red to Green)
- Eliminate specific color channels (e.g., remove Red)

### 📊 3. Histogram Operations
- Histogram Stretching (grayscale)
- Histogram Equalization (grayscale)

### 🧠 4. Neighborhood (Spatial Filtering)
- **Linear Filters**: Average, Laplacian
- **Non-linear Filters**: Maximum, Minimum, Median, Mode

### 🔧 5. Noise Reduction & Image Restoration
- **Salt & Pepper Noise**:
  - Average Filter
  - Median Filter
  - Outlier Method
- **Gaussian Noise**:
  - Image Averaging
  - Average Filter

### 🧩 6. Image Segmentation
- Basic Global Thresholding
- Automatic Thresholding
- Adaptive Thresholding

### 🧠 7. Edge Detection
- Sobel Edge Detector

### 🌐 8. Morphological Operations
- Dilation
- Erosion
- Opening
- Boundary Extraction:
  - Internal
  - External
  - Morphological Gradient

## 🖥️ Demo

To test the project locally:

```bash
git clone https://github.com/yourusername/dip-toolbox.git
cd dip-toolbox
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## 📂 Project Structure

```
dip-toolbox/
│
├── static/                 # CSS/JS/Images
├── templates/              # HTML templates
├── operations/             # Core processing functions
├── app.py                  # Main Flask app
├── requirements.txt        # Dependencies
└── README.md
```

## 📸 Screenshot

![Toolbox UI](./path-to-your-screenshot.png)

## ❤️ Made With

- Python
- Flask
- OpenCV
- NumPy
- HTML & CSS (Gradient UI)

## 👤 Author

**Abdalrahman Hossam Othman**  
📧 abdalrahman.hossam.othman@gmail.com  
🔗 [Portfolio](https://abdalrahmanothman01.github.io/AbdalrahmanOthman/)  
🔗 [LinkedIn](https://www.linkedin.com/in/abdalrahman-othman-)  
🔗 [GitHub](https://github.com/AbdalrahmanOthman01)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
