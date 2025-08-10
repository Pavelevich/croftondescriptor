#!/usr/bin/env python3
"""
Test the corrected CUDA-replicated algorithm directly
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os

class EnhancedEdgeDetector:
    def __init__(self):
        pass
    
    def enhanced_preprocessing(self, image):
        """Replicate the exact CUDA algorithm: HSV mask + Top-Hat + Morphology"""
        print("Starting enhanced preprocessing (CUDA replication)...")
        
        # 1) Convert to HSV for color-based mask (same as CUDA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        print(f"Converted to HSV: {hsv.shape}")
        
        # 2) Widen HSV range a bit (same as CUDA original)
        # Original CUDA: lowerPurple(100, 20, 20), upperPurple(180, 255, 255)
        lower_purple = np.array([100, 20, 20])
        upper_purple = np.array([180, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_purple, upper_purple)
        print(f"HSV mask created, white pixels: {cv2.countNonZero(mask_hsv)}")
        
        # 3) Top-Hat transform on grayscale to highlight faint edges (same as CUDA)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Morphological top-hat: top-hat = original - open(original)
        # This highlights bright regions smaller than structuring element
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_tophat)
        tophat = cv2.subtract(img_gray, opened_gray)  # highlight bright structures
        print(f"Top-hat transform applied with 15x15 kernel")
        
        # 4) Binarize top-hat with Otsu (same as CUDA)
        _, bin_tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print(f"Top-hat binarized, white pixels: {cv2.countNonZero(bin_tophat)}")
        
        # 5) Combine HSV mask with top-hat result (logical OR) - same as CUDA
        # This captures both color-based region AND faint grayscale edges
        combined = cv2.bitwise_or(mask_hsv, bin_tophat)
        print(f"Combined HSV+TopHat, white pixels: {cv2.countNonZero(combined)}")
        
        # 6) Morphology on combined (two-step: open then close) - same as CUDA
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        print(f"Final morphology - White pixels after open: {cv2.countNonZero(opened)}, after close: {cv2.countNonZero(closed)}")
        
        return closed, mask_hsv, bin_tophat, combined, opened

def test_algorithm():
    print("Testing Corrected CUDA-Replicated Algorithm")
    print("==========================================")
    
    # Load the test image
    image_path = "/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/test_cell.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return
        
    print(f"Loaded image: {image.shape}")
    
    # Create detector
    detector = EnhancedEdgeDetector()
    
    # Apply preprocessing
    preprocessed, mask_hsv, bin_tophat, combined, opened = detector.enhanced_preprocessing(image)
    
    # Find contours
    print("\nFinding contours...")
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("ERROR: No contours found!")
        return
    
    print(f"Found {len(contours)} contours")
    
    # 7) Largest contour (exact same as CUDA)
    largest_idx = 0
    largest_area = 0.0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_idx = i
    
    largest_contour = contours[largest_idx]
    print(f"Largest contour: area={largest_area}, perimeter={cv2.arcLength(largest_contour, True)}")
    
    # 8) Log average HSV in largest contour (same as CUDA)
    total_h, total_s, total_v = 0, 0, 0
    count_hsv = 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for pt in largest_contour:
        x, y = pt[0]
        if 0 <= x < hsv.shape[1] and 0 <= y < hsv.shape[0]:
            hsv_pixel = hsv[y, x]
            total_h += hsv_pixel[0]
            total_s += hsv_pixel[1] 
            total_v += hsv_pixel[2]
            count_hsv += 1
    
    if count_hsv > 0:
        avg_h = total_h / count_hsv
        avg_s = total_s / count_hsv
        avg_v = total_v / count_hsv
        print(f"Average H,S,V in largestContour = {avg_h:.1f}, {avg_s:.1f}, {avg_v:.1f}")
    else:
        print("Could not compute average HSV (count=0)")
    
    # Save visualization results
    print("\nSaving results...")
    
    # Original image
    cv2.imwrite("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/debug_original.jpg", image)
    
    # HSV mask
    cv2.imwrite("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/debug_hsv_mask.jpg", mask_hsv)
    
    # Top-hat
    cv2.imwrite("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/debug_tophat.jpg", bin_tophat)
    
    # Combined
    cv2.imwrite("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/debug_combined.jpg", combined)
    
    # After morphology
    cv2.imwrite("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/debug_preprocessed.jpg", preprocessed)
    
    # Final result with contour
    result_image = image.copy()
    cv2.drawContours(result_image, contours, largest_idx, (0, 255, 0), 2)
    cv2.imwrite("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/debug_final_result.jpg", result_image)
    
    print("Debug images saved:")
    print("- debug_original.jpg")
    print("- debug_hsv_mask.jpg") 
    print("- debug_tophat.jpg")
    print("- debug_combined.jpg")
    print("- debug_preprocessed.jpg")
    print("- debug_final_result.jpg")
    
    print(f"\nSUMMARY:")
    print(f"========")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"HSV mask pixels: {cv2.countNonZero(mask_hsv)}")
    print(f"Top-hat pixels: {cv2.countNonZero(bin_tophat)}")
    print(f"Combined pixels: {cv2.countNonZero(combined)}")
    print(f"Final preprocessed pixels: {cv2.countNonZero(preprocessed)}")
    print(f"Contours found: {len(contours)}")
    print(f"Largest contour area: {largest_area}")
    
    print("\n✓ Test completed successfully!")
    print("The corrected algorithm now follows the exact CUDA approach.")

if __name__ == "__main__":
    test_algorithm()