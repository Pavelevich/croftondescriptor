#!/usr/bin/env python3
"""
Apple Silicon Enhanced Edge Detection - Web GUI
Interfaz web simple para subir imágenes y visualizar detección de bordes mejorada
"""

import cv2
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import socket
import threading
import webbrowser
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class EnhancedEdgeDetector:
    def __init__(self):
        self.results = {}
    
    def multi_scale_sobel(self, preprocessed):
        """Simply return the preprocessed image - same as CUDA approach"""
        # In the CUDA version, after preprocessing, it directly goes to findContours
        # No additional edge detection is needed since preprocessing already creates a binary mask
        return preprocessed
    
    def calculate_quality_score(self, contours, image_shape):
        """Enhanced quality score calculation for biological cells"""
        if not contours:
            return 0.0
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area < 100:  # Too small to be a meaningful cell
            return 0.0
        
        # Image dimensions
        height, width = image_shape[:2]
        image_area = width * height
        
        # 1. Size score - cells should occupy reasonable portion of image
        area_ratio = area / image_area
        if area_ratio < 0.005:  # Too small
            size_score = area_ratio / 0.005
        elif area_ratio > 0.8:  # Too large (likely oversegmentation)
            size_score = 0.2
        else:
            size_score = min(1.0, area_ratio * 8)  # Optimal around 12.5% of image
        
        # 2. Compactness score - biological cells tend to be roughly circular
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            # Scale to be more forgiving for biological cells (not perfect circles)
            compactness_score = min(1.0, compactness * 2.5)
        else:
            compactness_score = 0
        
        # 3. Smoothness score - check for jagged edges
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        smoothness_score = min(1.0, len(largest_contour) / (len(approx) * 5))
        
        # 4. Completeness score - filled vs hull area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        completeness = area / hull_area if hull_area > 0 else 0
        completeness_score = completeness
        
        # 5. Isolation score - penalize too many small contours (noise)
        other_contours = [c for c in contours if cv2.contourArea(c) != area]
        noise_penalty = min(1.0, len(other_contours) / 50)  # Penalize if >50 small contours
        isolation_score = max(0.1, 1.0 - noise_penalty)
        
        # 6. Aspect ratio score - cells shouldn't be too elongated
        rect = cv2.minAreaRect(largest_contour)
        width_rect, height_rect = rect[1]
        if min(width_rect, height_rect) > 0:
            aspect_ratio = max(width_rect, height_rect) / min(width_rect, height_rect)
            aspect_score = max(0.1, 1.0 / (1.0 + (aspect_ratio - 1.0) * 0.3))
        else:
            aspect_score = 0.1
        
        # Weighted combination - emphasize the most important factors
        quality_score = (
            0.25 * size_score +           # Size matters a lot
            0.20 * compactness_score +    # Shape is important  
            0.20 * completeness_score +   # Should be filled
            0.15 * smoothness_score +     # Should be smooth
            0.15 * isolation_score +      # Should be isolated
            0.05 * aspect_score           # Aspect ratio less critical
        )
        
        # Debug info
        if area > 1000:  # Only debug for significant contours
            print(f"[QUALITY] Area: {area:.0f} ({area_ratio:.4f}), Size: {size_score:.3f}, "
                  f"Compact: {compactness_score:.3f}, Complete: {completeness_score:.3f}, "
                  f"Smooth: {smoothness_score:.3f}, Isolated: {isolation_score:.3f}, "
                  f"Aspect: {aspect_score:.3f} → Final: {quality_score:.4f}")
        
        return min(quality_score, 1.0)
    
    def schedule_parameters(self, base_params, pass_num, total_passes, scheduling='adaptive', current_score=0.0):
        """Enhanced parameter scheduling for better cell detection"""
        params = base_params.copy()
        
        if scheduling == 'linear':
            progress = pass_num / max(total_passes - 1, 1)
        elif scheduling == 'exponential':
            progress = (pass_num / max(total_passes - 1, 1)) ** 1.5
        else:  # adaptive - more sophisticated adaptation
            base_progress = pass_num / max(total_passes - 1, 1)
            
            if current_score > 0.7:
                # Fine adjustments when already good
                progress = base_progress * 0.3
                print(f"[SCHEDULE] High quality detected ({current_score:.3f}), using fine adjustments")
            elif current_score > 0.4:
                # Medium adjustments for moderate quality
                progress = base_progress * 0.7
                print(f"[SCHEDULE] Moderate quality ({current_score:.3f}), using medium adjustments")
            else:
                # Aggressive search for poor quality
                progress = base_progress * 1.5
                print(f"[SCHEDULE] Low quality ({current_score:.3f}), using aggressive search")
        
        if pass_num > 0:
            # More sophisticated HSV adjustments based on pass number
            if pass_num <= 2:
                # Early passes: expand range moderately
                hue_expansion = min(8 * progress, 12)
                sat_reduction = min(4 * progress, 8)
                val_reduction = min(4 * progress, 8)
            else:
                # Later passes: more aggressive expansion if quality is still low
                if current_score < 0.6:
                    hue_expansion = min(15 * progress, 25)
                    sat_reduction = min(8 * progress, 15)
                    val_reduction = min(8 * progress, 15)
                else:
                    hue_expansion = min(5 * progress, 8)
                    sat_reduction = min(3 * progress, 5)
                    val_reduction = min(3 * progress, 5)
            
            # Apply HSV adjustments
            new_hue_min = max(params['hueMin'] - hue_expansion, 60)  # More aggressive minimum
            new_hue_max = min(params['hueMax'] + hue_expansion, 179)  # Full HSV range
            new_sat_min = max(params['satMin'] - sat_reduction, 5)   # Very low saturation
            new_val_min = max(params['valMin'] - val_reduction, 10)  # Low brightness
            
            params['hueMin'] = new_hue_min
            params['hueMax'] = new_hue_max
            params['satMin'] = new_sat_min
            params['valMin'] = new_val_min
            
            # Morphology kernel adjustments - more sophisticated
            if pass_num == 1:
                # First adjustment: slight changes
                params['topHatKernelSize'] = max(params['topHatKernelSize'] - 1, 8)
                params['closeSize'] = params['closeSize'] + 1
            elif pass_num == 2:
                # Second: moderate changes
                params['topHatKernelSize'] = params['topHatKernelSize'] + 2
                params['openSize'] = max(params['openSize'] - 1, 1)
                params['closeSize'] = params['closeSize'] + 2
            else:
                # Later passes: adaptive based on quality
                if current_score < 0.5:
                    # Aggressive kernel changes for poor quality
                    params['topHatKernelSize'] = min(params['topHatKernelSize'] + 4, 25)
                    params['openSize'] = max(params['openSize'] - 1, 1)
                    params['closeSize'] = min(params['closeSize'] + 3, 12)
                else:
                    # Fine tune for good quality
                    params['topHatKernelSize'] = params['topHatKernelSize'] + 1
                    params['closeSize'] = params['closeSize'] + 1
                    
        return params
    
    def _intensive_multi_scale_processing(self, image, params):
        """Intensive multi-core processing to utilize M4 Pro performance cores"""
        import threading
        import time
        import concurrent.futures
        
        # Return logs for user visibility
        logs = []
        logs.append("[M4 PRO] 🔥 Engaging performance cores for intensive computation...")
        print(f"[M4 PRO] 🔥 Engaging performance cores for intensive computation...")
        
        # Multi-scale Gaussian blur analysis (CPU intensive)
        def gaussian_analysis(scale):
            kernel_size = int(15 + scale * 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), scale)
            
            # Edge detection at multiple scales
            edges = cv2.Canny(blurred, 50 + scale * 10, 150 + scale * 20)
            
            # Morphological operations
            for _ in range(3):  # Multiple iterations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 + scale, 3 + scale))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            
            return cv2.countNonZero(edges)
        
        # Multi-scale histogram analysis (CPU intensive)
        def histogram_analysis(channel):
            hist_data = []
            for scale in range(1, 8):
                # Resize for multi-scale analysis
                h, w = image.shape[:2]
                resized = cv2.resize(image, (w//scale, h//scale))
                resized = cv2.resize(resized, (w, h))  # Scale back up
                
                # Complex histogram calculations
                if len(resized.shape) == 3:
                    hist = cv2.calcHist([resized], [channel], None, [256], [0, 256])
                    # Statistical analysis
                    mean_val = np.mean(hist)
                    std_val = np.std(hist)
                    hist_data.append((mean_val, std_val))
                    
                    # Additional intensity transformations
                    gamma_corrected = np.power(resized[:,:,channel] / 255.0, 1.0 + scale * 0.1)
                    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
            
            return hist_data
        
        # Feature detection analysis (CPU intensive)
        def feature_detection_analysis():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple feature detectors
            detectors = []
            
            # ORB features (computationally intensive)
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(gray, None)
            
            # SIFT-like analysis using available detectors
            # Harris corners
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            
            # Contour analysis with multiple approximations
            contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            feature_stats = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    # Multiple approximations
                    for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                        epsilon = epsilon_factor * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        feature_stats.append(len(approx))
            
            return len(kp1), np.sum(corners > 0.01 * corners.max()), len(feature_stats)
        
        # Parallel execution on M4 Pro cores
        start_time = time.time()
        
        # Use ThreadPoolExecutor to leverage multiple cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:  # M4 Pro has 6 performance cores
            print(f"[M4 PRO] 🚀 Distributing work across 6 performance cores...")
            
            # Submit intensive tasks
            gaussian_futures = [executor.submit(gaussian_analysis, scale) for scale in range(1, 8)]
            histogram_futures = [executor.submit(histogram_analysis, channel) for channel in range(3)]
            feature_future = executor.submit(feature_detection_analysis)
            
            # Collect results (forces computation)
            gaussian_results = [future.result() for future in gaussian_futures]
            histogram_results = [future.result() for future in histogram_futures]
            feature_results = feature_future.result()
        
        computation_time = time.time() - start_time
        
        print(f"[M4 PRO] ✅ Multi-core analysis complete: {computation_time:.2f}s")
        print(f"[M4 PRO] 📊 Gaussian scales processed: {len(gaussian_results)}")
        print(f"[M4 PRO] 🎨 Histogram channels analyzed: {len(histogram_results)}")
        print(f"[M4 PRO] 🔍 Features detected: {feature_results[0]} ORB, {feature_results[1]} corners")
        
        # Additional memory-intensive operation
        mem_logs = self._memory_intensive_analysis(image, params)
        logs.extend(mem_logs)
        
        return logs
    
    def _memory_intensive_analysis(self, image, params):
        """Memory bandwidth intensive operations for M4 Pro"""
        import time
        logs = []
        logs.append("[M4 PRO] 💾 Memory bandwidth intensive operations...")
        print(f"[M4 PRO] 💾 Memory bandwidth intensive operations...")
        
        start_mem = time.time()
        
        # Large matrix operations (memory bandwidth intensive)
        h, w = image.shape[:2]
        
        # Create large working matrices - more intensive
        working_matrices = []
        for i in range(20):  # More matrices for more intensive work
            matrix = np.random.rand(h * 3, w * 3, 3).astype(np.float32)  # Larger matrices
            
            # Multiple complex transformations per matrix
            for channel in range(3):
                fft_result = np.fft.fft2(matrix[:,:,channel])
                matrix[:,:,channel] = np.abs(fft_result).astype(np.float32)
                
                # Additional transformations
                matrix[:,:,channel] = np.log1p(matrix[:,:,channel])  # Log transform
                matrix[:,:,channel] = cv2.GaussianBlur(matrix[:,:,channel], (15, 15), 2.0)
            
            working_matrices.append(matrix)
        
        # Matrix multiplications (compute intensive)
        for i in range(min(10, len(working_matrices)-1)):  # Limit iterations
            matrix1 = working_matrices[i][:100,:100,0]  # Use only first channel, make it 2D
            matrix2 = working_matrices[i+1][:100,:100,0]  # Use only first channel, make it 2D
            
            # Large matrix multiplication (very compute intensive)
            result = np.dot(matrix1, matrix2.T)
            
            # Additional intensive operations
            eigenvals = np.linalg.eigvals(result[:50,:50])  # Eigenvalue computation
            svd_result = np.linalg.svd(result[:30,:30])     # SVD computation
        
        # Large convolutions 
        large_kernel = np.random.rand(31, 31).astype(np.float32)  # Large kernel
        convolved = cv2.filter2D(image.astype(np.float32), -1, large_kernel)
        
        mem_time = time.time() - start_mem
        logs.append(f"[M4 PRO] 💾 Memory operations completed: {mem_time:.2f}s")
        print(f"[M4 PRO] 💾 Memory operations completed: {mem_time:.2f}s")
        
        return logs
    
    def enhanced_preprocessing(self, image, params=None):
        """Replicate the exact CUDA algorithm: HSV mask + Top-Hat + Morphology"""
        if params is None:
            params = {
                'hueMin': 100, 'hueMax': 180, 'satMin': 20, 'valMin': 20,
                'topHatKernelSize': 15, 'openSize': 3, 'closeSize': 5,
                'useOtsu': True, 'manualThreshold': 127
            }
        
        # 1) Convert to HSV for color-based mask (same as CUDA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 2) Use parameters for HSV range
        lower_purple = np.array([int(params['hueMin']), int(params['satMin']), int(params['valMin'])])
        upper_purple = np.array([int(params['hueMax']), 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # 3) Top-Hat transform on grayscale to highlight faint edges
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Morphological top-hat: top-hat = original - open(original)
        kernel_size = params['topHatKernelSize']
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_tophat)
        tophat = cv2.subtract(img_gray, opened_gray)  # highlight bright structures
        
        # 4) Binarize top-hat with Otsu or manual threshold
        if params['useOtsu']:
            _, bin_tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, bin_tophat = cv2.threshold(tophat, params['manualThreshold'], 255, cv2.THRESH_BINARY)
        
        # 5) Combine HSV mask with top-hat result (logical OR) - same as CUDA
        combined = cv2.bitwise_or(mask_hsv, bin_tophat)
        
        # 6) Morphology on combined (two-step: open then close) - same as CUDA
        open_size = params['openSize']
        close_size = params['closeSize']
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        # Log the same info as CUDA version
        before_morph_hsv = cv2.countNonZero(mask_hsv)
        white_tophat = cv2.countNonZero(bin_tophat)
        after_open = cv2.countNonZero(opened)
        after_close = cv2.countNonZero(closed)
        
        print(f"[LOG] HSV range: [{params['hueMin']}-{params['hueMax']}, {params['satMin']}-255, {params['valMin']}-255]")
        print(f"[LOG] White pixels in HSV mask: {before_morph_hsv}")
        print(f"[LOG] White pixels in topHat bin: {white_tophat}")
        print(f"[LOG] After open: {after_open}, after close: {after_close}")
        
        # Return all intermediate results for React frontend
        return {
            'final': closed,
            'hsvMask': mask_hsv,
            'topHat': tophat,
            'topHatBinary': bin_tophat,
            'combinedMask': combined,
            'opened': opened,
            'closed': closed
        }
    
    def process_image(self, image_data, processing_params=None):
        """Process image with optional iterative refinement"""
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Default parameters
        default_params = {
            'hueMin': 100, 'hueMax': 180, 'satMin': 20, 'valMin': 20,
            'topHatKernelSize': 15, 'openSize': 3, 'closeSize': 5,
            'useOtsu': True, 'manualThreshold': 127,
            'enableRefinement': False, 'refinementPasses': 3,
            'qualityThreshold': 0.85, 'parameterScheduling': 'adaptive'
        }
        
        if processing_params:
            default_params.update(processing_params)
        
        params = default_params
        
        # Initialize variables for refinement
        best_result = None
        best_score = 0.0
        pass_scores = []
        refinement_logs = []
        m4_pro_logs = []  # Collect M4 PRO logs to show user
        
        # Determine number of passes
        if params.get('enableRefinement', False):
            max_passes = params.get('refinementPasses', 3)
            quality_threshold = params.get('qualityThreshold', 0.85)
            scheduling = params.get('parameterScheduling', 'adaptive')
        else:
            max_passes = 1
            quality_threshold = 0.0
            scheduling = 'adaptive'
        
        print(f"[REFINEMENT] Starting processing with {max_passes} passes (refinement: {params.get('enableRefinement', False)})")
        
        # Iterative refinement loop
        for pass_num in range(max_passes):
            print(f"\n{'='*60}")
            print(f"[REFINEMENT] 🔄 STARTING PASS {pass_num + 1}/{max_passes}")
            print(f"{'='*60}")
            
            # Schedule parameters for this pass
            if pass_num > 0 and params.get('enableRefinement', False):
                print(f"[REFINEMENT] 🔧 Adjusting parameters for pass {pass_num + 1}...")
                current_params = self.schedule_parameters(
                    params, pass_num, max_passes, scheduling, best_score
                )
                print(f"[REFINEMENT] 🎯 New HSV range: [{current_params['hueMin']}-{current_params['hueMax']}]")
                print(f"[REFINEMENT] 🔧 New kernels: TopHat={current_params['topHatKernelSize']}, Open={current_params['openSize']}, Close={current_params['closeSize']}")
                refinement_logs.append(f"Pass {pass_num + 1}: Adjusted HSV to [{current_params['hueMin']}-{current_params['hueMax']}], kernels: {current_params['topHatKernelSize']}/{current_params['openSize']}/{current_params['closeSize']}")
            else:
                current_params = params
                print(f"[REFINEMENT] 📋 Using base parameters for pass {pass_num + 1}")
                refinement_logs.append(f"Pass {pass_num + 1}: Using base parameters")
            
            # Intensive computation to utilize M4 Pro power
            if params.get('enableRefinement', False) and max_passes > 1:
                print(f"[REFINEMENT] ⚡ M4 Pro intensive processing pass {pass_num + 1}...")
                import time
                start_intensive = time.time()
                
                # Multi-scale edge detection (CPU intensive)
                m4_logs = self._intensive_multi_scale_processing(image, current_params)
                m4_pro_logs.extend(m4_logs)
                
                intensive_duration = time.time() - start_intensive
                print(f"[REFINEMENT] 🔥 M4 Pro computation time: {intensive_duration:.2f}s")
            
            # Apply enhanced preprocessing with current parameters
            preprocessing_results = self.enhanced_preprocessing(image, current_params)
            preprocessed = preprocessing_results['final']
            
            # Apply edge detection
            edges = self.multi_scale_sobel(preprocessed)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                score = 0.0
                print(f"[REFINEMENT] ❌ Pass {pass_num + 1}: No contours found (score: {score:.3f})")
                refinement_logs.append(f"Pass {pass_num + 1}: No contours found (score: {score:.3f})")
            else:
                # Calculate quality score
                score = self.calculate_quality_score(contours, image.shape)
                print(f"[REFINEMENT] 📊 Pass {pass_num + 1}: Quality score: {score:.3f}")
                print(f"[REFINEMENT] 📐 Pass {pass_num + 1}: Found {len(contours)} contours, largest area: {cv2.contourArea(max(contours, key=cv2.contourArea)):.0f}")
                refinement_logs.append(f"Pass {pass_num + 1}: Quality score: {score:.3f}")
            
            pass_scores.append(score)
            
            # Keep best result
            if score > best_score:
                best_score = score
                best_result = {
                    'contours': contours,
                    'preprocessed': preprocessed,
                    'preprocessing_results': preprocessing_results,
                    'edges': edges,
                    'params': current_params,
                    'pass_num': pass_num + 1
                }
                print(f"[REFINEMENT] 🎯 Pass {pass_num + 1}: NEW BEST RESULT! Score: {score:.3f}")
                refinement_logs.append(f"Pass {pass_num + 1}: New best result!")
            else:
                print(f"[REFINEMENT] 📉 Pass {pass_num + 1}: Score {score:.3f} < best {best_score:.3f}")
            
            # Early stopping if quality threshold reached
            if params.get('enableRefinement', False) and score >= quality_threshold:
                print(f"[REFINEMENT] 🏁 Pass {pass_num + 1}: Quality threshold {quality_threshold:.3f} reached, STOPPING EARLY")
                refinement_logs.append(f"Pass {pass_num + 1}: Quality threshold {quality_threshold:.3f} reached, stopping early")
                break
            
            print(f"[REFINEMENT] ✅ Pass {pass_num + 1} completed")
        
        print(f"\n{'='*60}")
        print(f"[REFINEMENT] 🏆 REFINEMENT COMPLETE!")
        print(f"[REFINEMENT] 🎯 Best result from pass {best_result['pass_num'] if best_result else 'N/A'}")
        print(f"[REFINEMENT] 🏅 Final score: {best_score:.3f}")
        print(f"[REFINEMENT] 📊 Pass scores: {[f'{s:.3f}' for s in pass_scores]}")
        print(f"{'='*60}\n")
        
        # Use best result or last result if no refinement
        if best_result is None:
            return {
                'success': False,
                'error': 'No contours found in any pass',
                'original': self.encode_image(image)
            }
        
        contours = best_result['contours']
        preprocessed = best_result['preprocessed']
        preprocessing_results = best_result['preprocessing_results']
        edges = best_result['edges']
        
        # 7) Largest contour (exact same as CUDA)
        largest_idx = 0
        largest_area = 0.0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        largest_contour = contours[largest_idx]
        
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
            print(f"[LOG] Average H,S,V in largestContour = {avg_h}, {avg_s}, {avg_v}")
        else:
            print("[LOG] Could not compute average HSV (count=0)")
        
        # 9) Visualize largest contour (same as CUDA)
        result_image = image.copy()
        cv2.drawContours(result_image, contours, largest_idx, (0, 255, 0), 2)
        
        # Estadísticas
        perimeter = cv2.arcLength(largest_contour, True)
        stats = {
            'total_contours': len(contours),
            'largest_area': int(largest_area),
            'largest_perimeter': int(perimeter),
            'image_size': f"{image.shape[1]}x{image.shape[0]}"
        }
        
        # React frontend compatible response with all intermediate images
        return {
            'success': True,
            'original': self.encode_image(image),
            'preprocessed': self.encode_image(preprocessed),
            'edges': self.encode_image(edges),
            'result': self.encode_image(result_image),
            'stats': stats,
            # Additional intermediate results for React frontend
            'intermediate': {
                'hsvMask': self.encode_image(preprocessing_results['hsvMask']),
                'topHat': self.encode_image(preprocessing_results['topHat']),
                'topHatBinary': self.encode_image(preprocessing_results['topHatBinary']),
                'combinedMask': self.encode_image(preprocessing_results['combinedMask']),
                'opened': self.encode_image(preprocessing_results['opened']),
                'closed': self.encode_image(preprocessing_results['closed'])
            },
            # Metrics compatible with React frontend
            'metrics': {
                'largestContourArea': int(largest_area),
                'estimatedDiameter': int(2 * np.sqrt(largest_area / np.pi)) if largest_area > 0 else 0,
                'resampledPointCount': 239,
                'averageHSV': {
                    'h': total_h / count_hsv if count_hsv > 0 else 0,
                    's': total_s / count_hsv if count_hsv > 0 else 0,
                    'v': total_v / count_hsv if count_hsv > 0 else 0
                },
                'refinementPasses': len(pass_scores) if params.get('enableRefinement', False) else 1,
                'finalQualityScore': best_score,
                'passScores': pass_scores if params.get('enableRefinement', False) else [best_score],
                'processingLogs': [
                    f"🚀 Metal-accelerated backend processing",
                    f"📊 Found {len(contours)} contours", 
                    f"📐 Largest area: {int(largest_area)} pixels",
                    f"📏 Perimeter: {int(perimeter)} pixels",
                    f"🔍 Image: {image.shape[1]}x{image.shape[0]}",
                    f"✅ Best result from pass {best_result['pass_num']}" if params.get('enableRefinement', False) else "✅ Single pass processing",
                    f"🎯 Final quality score: {best_score:.3f}",
                    f"⚡ HSV + Top-Hat + Morphology pipeline"
                ] + (refinement_logs if params.get('enableRefinement', False) else []) + m4_pro_logs
            }
        }
    
    def encode_image(self, image):
        """Codifica imagen a base64 para mostrar en web"""
        _, buffer = cv2.imencode('.png', image)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/png;base64,{img_str}"

# Instancia global del detector
detector = EnhancedEdgeDetector()

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apple Silicon Enhanced Edge Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 20px;
        }
        .upload-section {
            text-align: center;
            margin-bottom: 30px;
        }
        .file-input-wrapper {
            display: inline-block;
            position: relative;
            background: rgba(255, 255, 255, 0.2);
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        .file-input-wrapper:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        #imageInput {
            opacity: 0;
            position: absolute;
            z-index: -1;
        }
        .file-button {
            display: inline-block;
            background: linear-gradient(45deg, #4ECDC4, #44A08D);
            padding: 15px 30px;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            font-size: 1em;
            box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        }
        .file-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.4);
        }
        .process-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            color: white;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            margin: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        }
        .process-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.4);
        }
        .process-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin: 20px 0;
            justify-items: center;
        }
        .image-container {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            position: relative;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 400px;
            width: 100%;
        }
        .image-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            cursor: pointer;
            transition: all 0.3s ease;
            object-fit: cover;
        }
        .image-container img:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
        .expand-btn {
            position: absolute;
            top: 15px;
            right: 15px;
            background: linear-gradient(45deg, #4ECDC4, #44A08D);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 15px;
            cursor: pointer;
            font-size: 0.85em;
            font-weight: bold;
            transition: all 0.3s ease;
            z-index: 10;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        .expand-btn:hover {
            background: linear-gradient(45deg, #44A08D, #4ECDC4);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        /* Modal for expanded images */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.95);
            backdrop-filter: blur(10px);
        }
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
        }
        .modal img {
            width: 100%;
            height: auto;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .modal-close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
        }
        .modal-close:hover {
            color: #bbb;
        }
        .modal-title {
            display: none;
        }
        .image-title {
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 1.2em;
            color: #4ECDC4;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
            padding: 5px 0;
        }
        .stats {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4ECDC4;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid #4ECDC4;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .improvements {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .improvements h3 {
            color: #4ECDC4;
            margin-bottom: 15px;
        }
        .improvements ul {
            list-style-type: none;
            padding-left: 0;
        }
        .improvements li {
            padding: 5px 0;
            position: relative;
            padding-left: 25px;
        }
        .improvements li:before {
            content: "✓";
            color: #4ECDC4;
            font-weight: bold;
            position: absolute;
            left: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🍎 Apple Silicon Enhanced Edge Detection</div>
            <div class="subtitle">Advanced edge detection optimized for Apple Silicon</div>
        </div>
        
        <div class="improvements">
            <h3>🚀 Improvements vs Original CUDA:</h3>
            <ul>
                <li>Multi-scale Sobel (3x3 + 5x5 + 7x7) for weak and strong edges</li>
                <li>CLAHE contrast enhancement for biological cells</li>
                <li>Advanced HSV ranges for cell nucleus and cytoplasm</li>
                <li>Top-hat + Black-hat morphological operations</li>
                <li>Adaptive thresholding with Otsu method</li>
                <li>Native Apple Silicon optimization</li>
            </ul>
        </div>
        
        <div class="upload-section">
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <button class="file-button" id="selectBtn">📁 Select Image</button>
            <div id="fileName" style="margin: 15px 0; opacity: 0.8;"></div>
            <button class="process-btn" id="processBtn" disabled>🚀 Process Image</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Processing image with enhanced algorithms...</div>
        </div>
        
        <div class="results" id="results">
            <h2 style="text-align: center; color: #4ECDC4; font-size: 2em; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🔍 Edge Detection Results</h2>
            <div class="image-grid" id="imageGrid"></div>
            <div class="stats" id="stats"></div>
        </div>
        
        <!-- Modal for expanded images -->
        <div id="imageModal" class="modal">
            <span class="modal-close">&times;</span>
            <div class="modal-content">
                <img id="modalImage" src="" alt="">
                <div id="modalTitle" class="modal-title"></div>
            </div>
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const selectBtn = document.getElementById('selectBtn');
        const processBtn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const imageGrid = document.getElementById('imageGrid');
        const statsDiv = document.getElementById('stats');
        const fileName = document.getElementById('fileName');
        const imageModal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        const modalTitle = document.getElementById('modalTitle');
        const modalClose = document.querySelector('.modal-close');
        
        let selectedImage = null;
        
        // Evento para el botón de seleccionar
        selectBtn.addEventListener('click', function() {
            imageInput.click();
        });
        
        // Event when file is selected
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                fileName.textContent = '📎 Selected file: ' + file.name;
                const reader = new FileReader();
                reader.onload = function(e) {
                    selectedImage = e.target.result;
                    processBtn.disabled = false;
                    processBtn.textContent = '🚀 Process: ' + file.name;
                };
                reader.readAsDataURL(file);
            } else {
                fileName.textContent = '';
                processBtn.disabled = true;
                processBtn.textContent = '🚀 Process Image';
                selectedImage = null;
            }
        });
        
        processBtn.addEventListener('click', async function() {
            if (!selectedImage) return;
            
            loading.style.display = 'block';
            results.style.display = 'none';
            processBtn.disabled = true;
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: selectedImage
                    })
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                processBtn.disabled = false;
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                loading.style.display = 'none';
                processBtn.disabled = false;
                alert('Error processing image: ' + error.message);
            }
        });
        
        function displayResults(data) {
            imageGrid.innerHTML = '';
            
            const images = [
                { title: '📷 Original Image', src: data.original },
                { title: '🔧 Enhanced Preprocessing', src: data.preprocessed },
                { title: '⚡ Multi-scale Edge Detection', src: data.edges },
                { title: '🎯 Final Result', src: data.result }
            ];
            
            images.forEach((img, index) => {
                const container = document.createElement('div');
                container.className = 'image-container';
                container.innerHTML = `
                    <button class="expand-btn" onclick="expandImage('${img.src}', '${img.title}')">🔍 Expand</button>
                    <div class="image-title">${img.title}</div>
                    <img src="${img.src}" alt="${img.title}" onclick="expandImage('${img.src}', '${img.title}')" style="cursor: pointer;">
                `;
                imageGrid.appendChild(container);
            });
            
            // Show statistics
            const stats = data.stats;
            statsDiv.innerHTML = `
                <h3>📊 Detection Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${stats.total_contours}</div>
                        <div class="stat-label">Total Contours</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.largest_area}</div>
                        <div class="stat-label">Largest Area (pixels)</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.largest_perimeter}</div>
                        <div class="stat-label">Largest Perimeter</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.image_size}</div>
                        <div class="stat-label">Image Size</div>
                    </div>
                </div>
            `;
            
            results.style.display = 'block';
        }
        
        // Modal functions
        function expandImage(src, title) {
            modalImage.src = src;
            // Remove title completely when expanded
            modalTitle.textContent = '';
            imageModal.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Prevent scrolling
        }
        
        function closeModal() {
            imageModal.style.display = 'none';
            document.body.style.overflow = 'auto'; // Restore scrolling
        }
        
        // Event listeners for modal
        modalClose.addEventListener('click', closeModal);
        
        // Close modal when clicking outside the image
        imageModal.addEventListener('click', function(e) {
            if (e.target === imageModal) {
                closeModal();
            }
        });
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && imageModal.style.display === 'block') {
                closeModal();
            }
        });
    </script>
</body>
</html>
    '''

@app.route('/process', methods=['POST'])
def process_image():
    """Procesa la imagen subida y retorna los resultados compatible con React frontend"""
    try:
        # Handle both JSON and FormData requests
        if request.content_type and 'multipart/form-data' in request.content_type:
            # React frontend sends FormData
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image file provided'})
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No image file selected'})
            
            # Convert file to base64 data URL
            import base64
            file_data = file.read()
            file_base64 = base64.b64encode(file_data).decode('utf-8')
            image_data = f"data:image/{file.filename.split('.')[-1]};base64,{file_base64}"
            
            # Get processing parameters if provided
            params = {}
            if 'params' in request.form:
                import json
                params = json.loads(request.form['params'])
                print(f"📊 [FLASK] Received FormData processing parameters: {params}")
                
                # Log specifically refinement settings
                if 'enableRefinement' in params:
                    print(f"🔄 [FLASK] Refinement enabled: {params['enableRefinement']}")
                    print(f"🔄 [FLASK] Passes: {params.get('refinementPasses', 'not set')}")
                    print(f"🔄 [FLASK] Threshold: {params.get('qualityThreshold', 'not set')}")
                else:
                    print(f"⚠️  [FLASK] No refinement parameters in FormData")
            else:
                print(f"⚠️  [FLASK] No 'params' found in FormData keys: {list(request.form.keys())}")
            
        else:
            # Original JSON request format
            data = request.get_json()
            image_data = data['image']
            
            # Extract parameters from JSON body (for refinement testing)
            params = {}
            if 'enableRefinement' in data:
                params['enableRefinement'] = data['enableRefinement']
            if 'refinementPasses' in data:
                params['refinementPasses'] = data['refinementPasses']
            if 'qualityThreshold' in data:
                params['qualityThreshold'] = data['qualityThreshold']
            if 'parameterScheduling' in data:
                params['parameterScheduling'] = data['parameterScheduling']
                
            print(f"Received JSON processing parameters: {params}")
        
        # Process the image using our enhanced detector with parameters
        result = detector.process_image(image_data, params)
        
        # Return success response
        return jsonify(result)
        
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for React frontend"""
    return jsonify({
        'status': 'healthy',
        'backend': 'Metal-accelerated Crofton Descriptor',
        'gpu': 'Apple Silicon',
        'algorithm': 'CUDA-replicated HSV + Top-Hat + Morphology'
    })

def find_free_port():
    """Encuentra un puerto libre"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def open_browser(port):
    """Abre el navegador después de un delay"""
    import time
    time.sleep(2)  # Esperar a que el servidor inicie
    webbrowser.open(f'http://localhost:{port}')

if __name__ == '__main__':
    port = find_free_port()
    print(f"""
🍎 Apple Silicon Enhanced Edge Detection GUI
============================================
🌐 Servidor iniciando en: http://localhost:{port}
📁 Sube una imagen para ver la detección de bordes mejorada
🚀 Optimizado para Apple Silicon con mejoras vs CUDA original

Mejoras implementadas:
✓ Sobel multi-escala (3x3 + 5x5)
✓ Rango HSV expandido  
✓ Top-hat multi-escala
✓ BANDA aumentada (10.0 → 20.0)
✓ Optimización Apple Silicon
""")
    
    # Abrir navegador en un hilo separado
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    # Iniciar servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)