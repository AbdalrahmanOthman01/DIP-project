import cv2
import numpy as np
from scipy import stats # Still used for mode filter
from PIL import Image
import math # For Gaussian kernel and rotation

# === Utility Functions ===

def cv2_to_pil(img_cv2): # Renamed img to img_cv2 to avoid conflict
    if img_cv2 is None:
        return None
    if len(img_cv2.shape) == 2: # Grayscale
        img_rgb = np.stack((img_cv2,) * 3, axis=-1) # Convert grayscale to RGB for PIL
    elif img_cv2.shape[2] == 3: # BGR
        img_rgb = img_cv2[..., ::-1] # BGR to RGB
    elif img_cv2.shape[2] == 1: # Grayscale with channel dim
        img_rgb = np.concatenate([img_cv2]*3, axis=-1)
    else: # Already RGB or other format PIL might handle
        img_rgb = img_cv2
    return Image.fromarray(img_rgb.astype(np.uint8))


def manual_bgr_to_gray(img_bgr):
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2: # Already grayscale
        return img_bgr
    # Standard weights: R=0.299, G=0.587, B=0.114
    # OpenCV uses BGR, so B=img[...,0], G=img[...,1], R=img[...,2]
    return np.dot(img_bgr[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

def manual_gray_to_bgr(img_gray):
    if img_gray is None:
        return None
    if len(img_gray.shape) == 3: # Already BGR (or color)
        return img_gray
    return np.stack((img_gray,) * 3, axis=-1)

def to_gray(img): # Keep this name for consistency in other functions
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 3:
        return manual_bgr_to_gray(img)
    return img # Assuming it's already grayscale or has a single channel

def manual_pad(image, pad_width, mode='constant', constant_values=0):
    """
    Manually pads an image. Simpler than np.pad for specific cases.
    pad_width is (top, bottom, left, right) or int for symmetric.
    Only supports 2D (grayscale) or 3D (color) images.
    """
    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width, pad_width, pad_width)
    
    h, w = image.shape[:2]
    new_h = h + pad_width[0] + pad_width[1]
    new_w = w + pad_width[2] + pad_width[3]

    if len(image.shape) == 3:
        c = image.shape[2]
        padded_image = np.full((new_h, new_w, c), constant_values, dtype=image.dtype)
        padded_image[pad_width[0]:pad_width[0]+h, pad_width[2]:pad_width[2]+w, :] = image
    else: # Grayscale
        padded_image = np.full((new_h, new_w), constant_values, dtype=image.dtype)
        padded_image[pad_width[0]:pad_width[0]+h, pad_width[2]:pad_width[2]+w] = image
    
    # Basic edge padding (replication)
    if mode == 'edge' or mode == 'reflect': # Simplified, 'reflect' is harder
        # Top
        padded_image[0:pad_width[0], pad_width[2]:pad_width[2]+w] = image[0:1, :]
        # Bottom
        padded_image[pad_width[0]+h:, pad_width[2]:pad_width[2]+w] = image[h-1:h, :]
        # Left
        padded_image[pad_width[0]:pad_width[0]+h, 0:pad_width[2]] = image[:, 0:1]
        # Right
        padded_image[pad_width[0]:pad_width[0]+h, pad_width[2]+w:] = image[:, w-1:w]
        # Corners (approximate)
        padded_image[0:pad_width[0], 0:pad_width[2]] = image[0,0]
        padded_image[0:pad_width[0], pad_width[2]+w:] = image[0,w-1]
        padded_image[pad_width[0]+h:, 0:pad_width[2]] = image[h-1,0]
        padded_image[pad_width[0]+h:, pad_width[2]+w:] = image[h-1,w-1]
        
    return padded_image


def manual_convolve2d(image, kernel):
    """
    Manual 2D convolution for a single channel image.
    Assumes kernel is smaller than image and has odd dimensions.
    Uses zero padding.
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # For simplicity, using np.pad here. Could be replaced by manual_pad.
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image, dtype=np.float64) # Use float for intermediate sums
    img_h, img_w = image.shape

    for r in range(img_h):
        for c in range(img_w):
            # Region of interest in padded image
            roi = padded_image[r : r + k_h, c : c + k_w]
            output[r, c] = np.sum(roi * kernel)
            
    return output

def apply_filter_to_channels(img_bgr, filter_func, *args):
    """ Helper to apply a grayscale filter to each channel of a BGR image """
    if len(img_bgr.shape) == 2: # Already grayscale
        return filter_func(img_bgr, *args)
    
    b, g, r = img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2]
    b_filtered = filter_func(b, *args)
    g_filtered = filter_func(g, *args)
    r_filtered = filter_func(r, *args)
    
    # Clip and convert back to uint8
    b_filtered = np.clip(b_filtered, 0, 255).astype(np.uint8)
    g_filtered = np.clip(g_filtered, 0, 255).astype(np.uint8)
    r_filtered = np.clip(r_filtered, 0, 255).astype(np.uint8)
    
    return np.stack((b_filtered, g_filtered, r_filtered), axis=-1)


# === Point Operations ===

def point_add(img, value=50):
    # Ensure no overflow/underflow and stay within uint8 range
    # Convert to a larger int type for addition, then clip
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)

def point_subtract(img, value=50):
    return np.clip(img.astype(np.int16) - value, 0, 255).astype(np.uint8)

def point_divide(img, divisor=2):
    # Integer division, then clip. Float conversion ensures intermediate precision if needed.
    return np.clip(img.astype(np.float32) / divisor, 0, 255).astype(np.uint8)

def point_complement(img):
    # For uint8 images, 255 - pixel_value
    return 255 - img

# === Color Image Operations ===

def change_light(img_bgr, value=50):
    # Kept cv2.cvtColor for HSV as it's complex to implement manually
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    v_new = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    hsv_new = np.stack((h, s, v_new), axis=-1)
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def change_red(img_bgr, value=50): # Assumes BGR
    img2 = img_bgr.copy()
    # Red channel is at index 2 for BGR
    img2[..., 2] = np.clip(img2[..., 2].astype(np.int16) + value, 0, 255).astype(np.uint8)
    return img2

def swap_rg(img_bgr): # Assumes BGR, swaps Red and Green
    img2 = img_bgr.copy()
    # BGR: B=0, G=1, R=2
    # New G = Old R, New R = Old G
    g_channel = img2[..., 1].copy() # Store original Green
    img2[..., 1] = img2[..., 2]     # G becomes R
    img2[..., 2] = g_channel        # R becomes original G
    return img2

def eliminate_red(img_bgr): # Assumes BGR
    img2 = img_bgr.copy()
    img2[..., 2] = 0 # Set Red channel to 0
    return img2

# === Histogram Operations ===

def hist_stretch(img):
    gray = to_gray(img)
    if gray is None: return None
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    if max_val == min_val: # Avoid division by zero if image is flat
        stretched = gray 
    else:
        stretched = ((gray - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
    return manual_gray_to_bgr(stretched)

def manual_hist_equalize_gray(gray_img):
    if gray_img is None: return None
    # 1. Compute histogram
    hist = np.zeros(256, dtype=int)
    for pixel_val in gray_img.flat:
        hist[pixel_val] += 1
    
    # 2. Compute CDF
    cdf = np.zeros(256, dtype=float)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
        
    # 3. Normalize CDF to 0-255 range
    # cdf_m = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
    # Find first non-zero cdf value for cdf_min
    cdf_min = 0
    for val in cdf:
        if val > 0:
            cdf_min = val
            break
            
    total_pixels = gray_img.size
    if total_pixels == cdf_min: # All pixels are the same, or only one non-zero CDF value at the end
         # Handle flat image or image where all pixels are below the first non-zero cdf value
        scale_factor = 0 # This will map everything to 0 if cdf_min is total_pixels
        if (total_pixels - cdf_min) > 0 :
             scale_factor = 255.0 / (total_pixels - cdf_min)
    else:
        scale_factor = 255.0 / (total_pixels - cdf_min)


    hist_eq_lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if cdf[i] >= cdf_min:
             hist_eq_lut[i] = np.clip(round((cdf[i] - cdf_min) * scale_factor), 0, 255)
        else:
            hist_eq_lut[i] = 0 # map values below cdf_min to 0
            
    # 4. Map pixel values using the LUT
    equalized_img = np.zeros_like(gray_img)
    for r in range(gray_img.shape[0]):
        for c in range(gray_img.shape[1]):
            equalized_img[r, c] = hist_eq_lut[gray_img[r, c]]
            
    return equalized_img.astype(np.uint8)

def hist_equalize(img):
    gray = to_gray(img)
    eq = manual_hist_equalize_gray(gray)
    return manual_gray_to_bgr(eq)

# === Spatial Filters ===

def _avg_filter_gray(gray_img, k_size=3):
    kernel = np.ones((k_size, k_size), dtype=np.float32) / (k_size * k_size)
    filtered = manual_convolve2d(gray_img.astype(np.float32), kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def avg_filter(img, k_size=3): # OpenCV blur uses (k_size, k_size)
    if img is None: return None
    return apply_filter_to_channels(img, _avg_filter_gray, k_size)


def _lap_filter_gray(gray_img):
    # Common Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    # Or kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # Convolve (needs to handle negative values correctly before abs and scaling)
    # OpenCV's CV_64F implies using float64 for precision during convolution
    lap = manual_convolve2d(gray_img.astype(np.float64), kernel)
    
    # Scale and convert (similar to cv2.convertScaleAbs)
    # Option 1: Simple absolute value (loses some info)
    # lap_abs = np.abs(lap)
    # lap_scaled = np.clip(lap_abs, 0, 255).astype(np.uint8)

    # Option 2: Scale to 0-255 range if min/max are known or fixed
    # For display, often good to scale.
    # Here, we mimic convertScaleAbs more closely by just taking abs and clipping
    # The true convertScaleAbs can also apply alpha and beta: result = alpha*src + beta
    lap_abs_scaled = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return lap_abs_scaled


def lap_filter(img):
    if img is None: return None
    gray = to_gray(img) # Laplacian is usually on grayscale
    lap_gray = _lap_filter_gray(gray)
    return manual_gray_to_bgr(lap_gray)


def _order_filter_gray(gray_img, k_size, operation):
    pad = k_size // 2
    padded_img = np.pad(gray_img, pad, mode='edge')
    output = np.zeros_like(gray_img)
    h, w = gray_img.shape

    for r in range(h):
        for c in range(w):
            window = padded_img[r : r + k_size, c : c + k_size]
            if operation == 'max':
                output[r, c] = np.max(window)
            elif operation == 'min':
                output[r, c] = np.min(window)
            elif operation == 'median':
                output[r, c] = np.median(window)
    return output.astype(np.uint8)

def max_filter(img, k_size=3): # cv2.dilate uses a structuring element
    if img is None: return None
    # Note: cv2.dilate with ones kernel is equivalent to max filter
    return apply_filter_to_channels(img, _order_filter_gray, k_size, 'max')

def min_filter(img, k_size=3): # cv2.erode uses a structuring element
    if img is None: return None
    # Note: cv2.erode with ones kernel is equivalent to min filter
    return apply_filter_to_channels(img, _order_filter_gray, k_size, 'min')

def median_filter(img, k_size=3):
    if img is None: return None
    return apply_filter_to_channels(img, _order_filter_gray, k_size, 'median')


def mode_filter(img, k_size=3): # Original mode_filter seems fine as it uses scipy.stats
    # This one is already fairly manual using scipy.stats.mode
    def modefilt2d(a, window_size):
        pad_size = window_size // 2
        padded = np.pad(a, pad_size, mode='edge') # Using np.pad for convenience
        out = np.zeros_like(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                mode_val = stats.mode(window.flatten(), keepdims=False)[0]
                out[i, j] = mode_val
        return out

    if img is None: return None
    if len(img.shape) == 2:
        return modefilt2d(img, k_size).astype(np.uint8)
    else:
        channels = [img[...,i] for i in range(img.shape[2])] # More general than cv2.split
        filtered_channels = [modefilt2d(ch, k_size) for ch in channels]
        return np.stack(filtered_channels, axis=-1).astype(np.uint8)


# === Noise Addition & Removal ===
# add_salt_pepper and add_gauss are already NumPy based, so they are fine.
# remove_sp_avg, remove_sp_median will use the manual avg_filter and median_filter above.

def add_salt_pepper(img): # This is already NumPy based, so it's fine.
    if img is None: return None
    s_vs_p = 0.5
    amount = 0.05
    out = img.copy()
    
    # Salt
    num_salt = np.ceil(amount * img.size * s_vs_p / img.shape[2] if len(img.shape)==3 else amount * img.size * s_vs_p) # Adjust for channels
    coords_salt_row = np.random.randint(0, img.shape[0], int(num_salt))
    coords_salt_col = np.random.randint(0, img.shape[1], int(num_salt))
    if len(img.shape) == 3:
        for i in range(img.shape[2]): # Apply to all channels
            out[coords_salt_row, coords_salt_col, i] = 255
    else:
        out[coords_salt_row, coords_salt_col] = 255
        
    # Pepper
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p) / img.shape[2] if len(img.shape)==3 else amount * img.size * (1. - s_vs_p))
    coords_pep_row = np.random.randint(0, img.shape[0], int(num_pepper))
    coords_pep_col = np.random.randint(0, img.shape[1], int(num_pepper))
    if len(img.shape) == 3:
        for i in range(img.shape[2]): # Apply to all channels
             out[coords_pep_row, coords_pep_col, i] = 0
    else:
        out[coords_pep_row, coords_pep_col] = 0
    return out

def remove_sp_avg(img):
    if img is None: return None
    return avg_filter(img, 3) # Uses our manual avg_filter

def remove_sp_median(img):
    if img is None: return None
    return median_filter(img, 3) # Uses our manual median_filter

def remove_sp_outlier(img_orig, threshold=50, k_size=3):
    if img_orig is None: return None
    result = img_orig.copy()
    
    # Median filter result to compare against
    # For color images, apply median per channel, then compare original gray to median gray
    if len(img_orig.shape) == 3:
        med_bgr = median_filter(img_orig, k_size) # Our manual median filter
        gray_orig = to_gray(img_orig)
        gray_med = to_gray(med_bgr) # Median of a BGR image, then to gray
    else: # Grayscale
        gray_orig = img_orig
        gray_med = median_filter(gray_orig, k_size) # Our manual median filter
        med_bgr = None # Not needed for grayscale assignment

    # Calculate difference
    # diff = np.abs(gray_orig.astype(np.int16) - gray_med.astype(np.int16)).astype(np.uint8)
    # Using np.absdiff is fine as it's a basic operation
    diff = np.zeros_like(gray_orig, dtype=np.uint8)
    for r in range(gray_orig.shape[0]):
        for c in range(gray_orig.shape[1]):
            d = int(gray_orig[r,c]) - int(gray_med[r,c])
            diff[r,c] = abs(d)

    mask = diff > threshold
    
    if len(img_orig.shape) == 3:
        # Replace noisy pixels in color image with corresponding pixels from median filtered color image
        for c_idx in range(3):
            result[..., c_idx][mask] = med_bgr[..., c_idx][mask]
    else:
        result[mask] = gray_med[mask]
    return result


def add_gauss(img, mean=0, sigma=25): # This is already NumPy based, fine.
    if img is None: return None
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def remove_gauss_avg(img):
    if img is None: return None
    return avg_filter(img, 3) # Uses our manual avg_filter

# === Thresholding & Segmentation ===

def _thresh_basic_gray(gray_img, threshold_val=127):
    # For each pixel, if > threshold_val, set to 255, else 0
    return ((gray_img > threshold_val) * 255).astype(np.uint8)

def thresh_basic(img, threshold_val=127):
    if img is None: return None
    gray = to_gray(img)
    th_gray = _thresh_basic_gray(gray, threshold_val)
    return manual_gray_to_bgr(th_gray)

def _otsu_threshold_gray(gray_img):
    # 1. Calculate normalized histogram
    hist = np.zeros(256, dtype=float)
    for pixel_val in gray_img.flat:
        hist[pixel_val] += 1
    hist_norm = hist / gray_img.size
    
    max_variance = 0
    best_thresh = 0
    
    for t in range(1, 256): # Iterate through all possible thresholds
        w0 = np.sum(hist_norm[:t])  # Probability of class 0 (background)
        w1 = np.sum(hist_norm[t:])  # Probability of class 1 (foreground)
        
        if w0 == 0 or w1 == 0:
            continue
            
        # Mean of class 0
        mu0 = np.sum(np.arange(t) * hist_norm[:t]) / w0
        # Mean of class 1
        mu1 = np.sum(np.arange(t, 256) * hist_norm[t:]) / w1
        
        # Inter-class variance
        variance = w0 * w1 * ((mu0 - mu1) ** 2)
        
        if variance > max_variance:
            max_variance = variance
            best_thresh = t
            
    return best_thresh

def thresh_auto(img): # Otsu's method
    if img is None: return None
    gray = to_gray(img)
    otsu_thresh_val = _otsu_threshold_gray(gray)
    th_gray = _thresh_basic_gray(gray, otsu_thresh_val)
    return manual_gray_to_bgr(th_gray)


def _thresh_adapt_gray(gray_img, block_size=11, C=2, method='mean'):
    # method can be 'mean' or 'gaussian' (gaussian is more complex to implement manually here)
    if block_size % 2 == 0:
        block_size += 1 # Must be odd
    
    pad = block_size // 2
    padded_gray = np.pad(gray_img, pad, mode='reflect') # Reflect padding is common
    thresh_img = np.zeros_like(gray_img)
    h, w = gray_img.shape

    for r in range(h):
        for c in range(w):
            window = padded_gray[r : r + block_size, c : c + block_size]
            
            if method == 'mean':
                local_thresh_val = np.mean(window) - C
            # elif method == 'gaussian': # Would need a Gaussian weighted average
            #     # Create Gaussian kernel for the window size
            #     # local_thresh_val = np.sum(window * gaussian_kernel) - C
            #     local_thresh_val = np.mean(window) - C # Placeholder for simplicity
            else: # Default to mean
                local_thresh_val = np.mean(window) - C

            if gray_img[r, c] > local_thresh_val:
                thresh_img[r, c] = 255
            else:
                thresh_img[r, c] = 0
    return thresh_img.astype(np.uint8)

def thresh_adapt(img):
    if img is None: return None
    gray = to_gray(img)
    # cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    th_gray = _thresh_adapt_gray(gray, block_size=11, C=2, method='mean')
    return manual_gray_to_bgr(th_gray)


# === Edge Detection ===

def _sobel_gray(gray_img):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)

    gx = manual_convolve2d(gray_img.astype(np.float32), kernel_x)
    gy = manual_convolve2d(gray_img.astype(np.float32), kernel_y)

    # Magnitude: sqrt(gx^2 + gy^2)
    # magnitude = np.sqrt(gx**2 + gy**2)
    # For performance, often approximated as |gx| + |gy|, but let's do sqrt
    magnitude = np.hypot(gx,gy) # hypot is element-wise sqrt(x1**2 + x2**2)
    
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude

def sobel(img):
    if img is None: return None
    gray = to_gray(img)
    sobel_gray_img = _sobel_gray(gray)
    return manual_gray_to_bgr(sobel_gray_img)

def _canny_simplified_gray(gray_img, low_thresh, high_thresh):
    # 1. Gaussian Blur (using our manual avg_filter as a simpler substitute or implement manual Gaussian)
    # blurred_gray = avg_filter(manual_gray_to_bgr(gray_img), 5) # Apply avg_filter, expects BGR
    # blurred_gray = to_gray(blurred_gray) # Convert back to gray
    # Let's use a manual Gaussian blur for better Canny-like behavior
    
    def create_gaussian_kernel(size=5, sigma=1.4):
        kernel = np.zeros((size, size))
        center = size // 2
        sum_val = 0
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
                sum_val += kernel[i, j]
        kernel = kernel / sum_val # Normalize
        return kernel

    gaussian_kernel = create_gaussian_kernel(5, 1.4) # Typical Canny params
    blurred_gray = manual_convolve2d(gray_img.astype(np.float32), gaussian_kernel)
    blurred_gray = np.clip(blurred_gray, 0, 255).astype(np.uint8)

    # 2. Sobel Operator (Gradient magnitude and direction)
    # We only need magnitude for a simplified version
    sobel_mag = _sobel_gray(blurred_gray) # Our manual Sobel

    # 3. Non-Maximum Suppression (Skipping for simplification - this is complex)
    # 4. Double Thresholding (Simplified - just apply high threshold for strong edges)
    #    A full Canny uses low_thresh for weak edges and connects them to strong edges.
    edges = np.zeros_like(sobel_mag)
    edges[sobel_mag >= high_thresh] = 255
    # A slightly better simplification:
    # strong_edges_r, strong_edges_c = np.where(sobel_mag >= high_thresh)
    # weak_edges_r, weak_edges_c = np.where((sobel_mag >= low_thresh) & (sobel_mag < high_thresh))
    # edges[strong_edges_r, strong_edges_c] = 255
    # For weak edges, one would check 8-connectivity to strong edges (hysteresis part - complex)
    
    return edges.astype(np.uint8)

def apply_edge_detection(img): # Canny
    if img is None: return None
    gray = to_gray(img)
    # cv2.Canny(to_gray(img), 100, 200)
    # Using simplified Canny due to complexity of NMS and Hysteresis
    edges_gray = _canny_simplified_gray(gray, 100, 200) 
    return manual_gray_to_bgr(edges_gray)

# === Morphological Operations ===
# These will use the _order_filter_gray with 'max' for dilate and 'min' for erode

def dilate(img, k_size=3):
    if img is None: return None
    # Dilation is max filter with a kernel of ones
    return apply_filter_to_channels(img, _order_filter_gray, k_size, 'max')

def erode(img, k_size=3):
    if img is None: return None
    # Erosion is min filter with a kernel of ones
    return apply_filter_to_channels(img, _order_filter_gray, k_size, 'min')

def opening(img, k_size=3):
    if img is None: return None
    # Opening is erosion then dilation
    eroded_img = erode(img, k_size)
    opened_img = dilate(eroded_img, k_size)
    return opened_img

def _boundary_internal_gray(gray_img, k_size=3):
    eroded_gray = _order_filter_gray(gray_img, k_size, 'min')
    # boundary = gray - eroded
    # Ensure no underflow:
    boundary = np.clip(gray_img.astype(np.int16) - eroded_gray.astype(np.int16), 0, 255).astype(np.uint8)
    return boundary

def boundary_internal(img, k_size=3):
    if img is None: return None
    # Usually applied on binary or grayscale images
    # If color, apply to grayscale version
    gray = to_gray(img) 
    boundary_gray = _boundary_internal_gray(gray, k_size)
    return manual_gray_to_bgr(boundary_gray)

def _boundary_external_gray(gray_img, k_size=3):
    dilated_gray = _order_filter_gray(gray_img, k_size, 'max')
    # boundary = dilated - gray
    boundary = np.clip(dilated_gray.astype(np.int16) - gray_img.astype(np.int16), 0, 255).astype(np.uint8)
    return boundary

def boundary_external(img, k_size=3):
    if img is None: return None
    gray = to_gray(img)
    boundary_gray = _boundary_external_gray(gray, k_size)
    return manual_gray_to_bgr(boundary_gray)

def _boundary_gradient_gray(gray_img, k_size=3):
    dilated_gray = _order_filter_gray(gray_img, k_size, 'max')
    eroded_gray = _order_filter_gray(gray_img, k_size, 'min')
    # gradient = dilated - eroded
    gradient = np.clip(dilated_gray.astype(np.int16) - eroded_gray.astype(np.int16), 0, 255).astype(np.uint8)
    return gradient

def boundary_gradient(img, k_size=3):
    if img is None: return None
    gray = to_gray(img)
    gradient_gray = _boundary_gradient_gray(gray, k_size)
    return manual_gray_to_bgr(gradient_gray)

# === Extra Operations ===

def _gaussian_blur_gray(gray_img, k_size=5, sigma=0): # sigma=0 means calculate from k_size
    if sigma == 0:
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8 # OpenCV formula
    
    kernel_1d = np.zeros(k_size)
    center = k_size // 2
    sum_val = 0
    for i in range(k_size):
        x = i - center
        kernel_1d[i] = np.exp(-(x**2) / (2 * sigma**2))
        sum_val += kernel_1d[i]
    kernel_1d /= sum_val # Normalize

    # Create 2D kernel from 1D (separable property)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    blurred = manual_convolve2d(gray_img.astype(np.float32), kernel_2d)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def apply_blur(img, k_size=5, sigma=0): # Gaussian Blur
    if img is None: return None
    # cv2.GaussianBlur(img, (5,5), 0)
    return apply_filter_to_channels(img, _gaussian_blur_gray, k_size, sigma)

def _sharpen_gray(gray_img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = manual_convolve2d(gray_img.astype(np.float32), kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
    
def apply_sharpen(img):
    if img is None: return None
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # return cv2.filter2D(img, -1, kernel)
    return apply_filter_to_channels(img, _sharpen_gray)


def apply_hist_eq(img): # This is a duplicate name, will use our hist_equalize
    return hist_equalize(img)


def manual_rotate_image(image, angle_degrees, center=None, interpolation='nearest'):
    """
    Manual rotation. This is complex to do well, especially with good interpolation.
    'nearest' neighbor interpolation implemented for simplicity. Bilinear is better but harder.
    Keeping cv2.warpAffine for production, this is for demonstration.
    """
    if image is None: return None
    rows, cols = image.shape[:2]
    
    if center is None:
        center_x, center_y = cols / 2, rows / 2
    else:
        center_x, center_y = center

    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Create output image (same size or bounding box size)
    # For simplicity, same size, parts will be cropped.
    rotated_image = np.zeros_like(image)

    for r_out in range(rows):
        for c_out in range(cols):
            # Translate to origin
            x = c_out - center_x
            y = r_out - center_y

            # Apply inverse rotation to find corresponding source pixel
            # x_src = x * cos_a + y * sin_a
            # y_src = -x * sin_a + y * cos_a
            # Corrected for mapping output to input (inverse transform)
            x_src = x * cos_a - y * sin_a
            y_src = x * sin_a + y * cos_a
            
            # Translate back
            x_src += center_x
            y_src += center_y

            if interpolation == 'nearest':
                r_src, c_src = round(y_src), round(x_src)
                if 0 <= r_src < rows and 0 <= c_src < cols:
                    if len(image.shape) == 3:
                        rotated_image[r_out, c_out, :] = image[r_src, c_src, :]
                    else:
                        rotated_image[r_out, c_out] = image[r_src, c_src]
            # elif interpolation == 'bilinear':
            #     # ... (more complex: get 4 nearest pixels, weighted average)
            #     pass
                
    return rotated_image.astype(image.dtype)


def apply_rotation(img):
    if img is None: return None
    # (h, w) = img.shape[:2]
    # M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
    # return cv2.warpAffine(img, M, (w, h))
    # Using manual rotation for demonstration (simple nearest neighbor)
    # For high quality, cv2.warpAffine is far superior.
    return manual_rotate_image(img, 45)
