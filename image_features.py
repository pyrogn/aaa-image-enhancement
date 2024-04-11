import cv2
import numpy as np
from skimage import exposure, color, img_as_float

def compute_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[...,2].mean()
    return brightness

def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def detect_blur(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm
    
def compute_color_distribution(image, bins=8):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm
    
def compute_dynamic_range(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.max(gray) - np.min(gray)

def detect_edges(image, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,threshold1,threshold2)
    return edges

# Load an example image
image = cv2.imread('example.jpg')

# Compute attributes
brightness = compute_brightness(image)
contrast = compute_contrast(image)
is_blurry, blur_score = detect_blur(image)  
color_dist = compute_color_distribution(image)
sharpness = compute_sharpness(image)
dynamic_range = compute_dynamic_range(image)
edges = detect_edges(image)

# Print results  
print(f"Brightness: {brightness:.2f}")
print(f"Contrast: {contrast:.2f}") 
print(f"Blurry: {is_blurry} (score: {blur_score:.2f})")
print(f"Color distribution: {color_dist}")
print(f"Sharpness: {sharpness:.2f}")  
print(f"Dynamic range: {dynamic_range}")
