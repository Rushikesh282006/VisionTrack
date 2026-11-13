
import sys
import torch
import cv2
import ultralytics

print("VISIONTRACK ENVIRONMENT CHECK")
print("-" * 40)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics:", ultralytics.__version__)

print("CUDA available:", torch.cuda.is_available())

print("Selected device:", "CPU")

print("-" * 40)
print("Environment verification completed!")