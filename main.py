import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, flash, session, redirect, url_for
from werkzeug.utils import secure_filename
import base64

from FN import *

app = Flask(__name__)
# Use absolute path for upload directory
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
app.secret_key = 'supersecret'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

topics = {
    "Point Operations": {
        "Addition": "point_add",
        "Subtraction": "point_subtract",
        "Division": "point_divide",
        "Complement": "point_complement",
    },
    "Color Operations": {
        "Change Lighting": "change_light",
        "Change Red": "change_red",
        "Swap R<->G": "swap_rg",
        "Eliminate Red": "eliminate_red",
    },
    "Histogram Operations": {
        "Histogram Stretch": "hist_stretch",
        "Histogram Equalize": "hist_equalize",
    },
    "Spatial Filters": {
        "Average Filter": "avg_filter",
        "Laplacian Filter": "lap_filter",
        "Max Filter": "max_filter",
        "Min Filter": "min_filter",
        "Median Filter": "median_filter",
        "Mode Filter": "mode_filter",
    },
    "Noise & Restoration": {
        "Add Salt & Pepper": "add_salt_pepper",
        "Remove SP Avg": "remove_sp_avg",
        "Remove SP Median": "remove_sp_median",
        "Remove SP Outlier": "remove_sp_outlier",
        "Add Gaussian": "add_gauss",
        "Remove Gaussian Avg": "remove_gauss_avg",
    },
    "Thresholding & Segmentation": {
        "Threshold Basic": "thresh_basic",
        "Threshold Auto": "thresh_auto",
        "Threshold Adaptive": "thresh_adapt",
    },
    "Edge Detection": {
        "Sobel Edge": "sobel",
        "Canny Edge": "apply_edge_detection",
    },
    "Morphological Operations": {
        "Dilate": "dilate",
        "Erode": "erode",
        "Opening": "opening",
        "Internal Boundary": "boundary_internal",
        "External Boundary": "boundary_external",
        "Morph Gradient": "boundary_gradient",
    },
    "Extra": {
        "Blur": "apply_blur",
        "Sharpen": "apply_sharpen",
        "Hist Eq": "apply_hist_eq",
        "Rotate": "apply_rotation",
    }  
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def cv2_to_base64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')

def get_function_name(selected_operation):
    for ops in topics.values():
        if selected_operation in ops:
            return ops[selected_operation]
    return None

fn_map = {fn: globals()[fn] for cat in topics.values() for fn in cat.values()}

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = result_image = None
    selected_operation = None

    if request.method == 'POST' and 'clear_image' in request.form:
        session.pop('uploaded_image_path', None)
        return redirect(url_for('index'))

    if request.method == 'POST' and 'operation' in request.form:
        file = request.files.get('image')
        selected_operation = request.form.get('operation')

        # Handle file upload
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Store absolute path in session
            session['uploaded_image_path'] = os.path.abspath(filepath)
        
        # Process operation
        img_path = session.get('uploaded_image_path')
        if img_path and os.path.exists(img_path) and selected_operation:
            img = cv2.imread(img_path)
            if img is not None:
                uploaded_image = "data:image/png;base64," + cv2_to_base64(img)
                fn_name = get_function_name(selected_operation)
                if fn_name:
                    try:
                        processed = fn_map[fn_name](img)
                        result_image = "data:image/png;base64," + cv2_to_base64(processed)
                        # Save result without affecting session path
                        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'result.png'), processed)
                    except Exception as e:
                        flash(f'Error processing image: {str(e)}', 'error')
                else:
                    flash('Invalid operation selected', 'error')
            else:
                flash('Failed to read uploaded image', 'error')
        elif not img_path:
            flash('Please upload an image first', 'error')

    # Show existing image if available
    img_path = session.get('uploaded_image_path')
    if img_path and os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            uploaded_image = "data:image/png;base64," + cv2_to_base64(img)

    return render_template('index.html',
                           topics=topics,
                           selected_operation=selected_operation,
                           uploaded_image=uploaded_image,
                           result_image=result_image)

@app.route('/download')
def download():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        flash('No processed image available', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)