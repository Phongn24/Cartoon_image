import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
  """Reads an image and displays it."""
  img = cv2.imread(filename)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.show()
  return img

def edge_mask(img, line_size, blur_value):
  """Detects edges in an image."""
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 225, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

def color_quantization(img, k):
  data = np.float32(img).reshape((-1, 3))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

def cartoon(img, edges):
  blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
  c = cv2.bitwise_and(blurred, blurred, mask=edges)
  plt.imshow(c)
  plt.show()

# Main execution
filename = "Dog.jpg"
img = read_file(filename)

line_size, blur_val = 3, 3
edges = edge_mask(img, line_size, blur_val)

img_quantized = color_quantization(img, 10)
blurred = cv2.bilateralFilter(img_quantized, d=9, sigmaColor=200, sigmaSpace=200)

cartoon(blurred, edges)