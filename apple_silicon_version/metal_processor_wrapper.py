#!/usr/bin/env python3
"""
Metal-Optimized Image Processor - Python Wrapper
Integrates Metal GPU acceleration with Flask backend
"""

import subprocess
import json
import tempfile
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import time

class MetalImageProcessor:
    """
    Python wrapper for Metal-optimized C++ image processor
    Provides seamless integration with Flask backend
    """

    def __init__(self, binary_path: Optional[str] = None):
        """
        Initialize Metal processor

        Args:
            binary_path: Path to crofton_optimized binary
        """
        if binary_path is None:
            # Try to find binary in common locations
            possible_paths = [
                "./build_optimized/crofton_optimized",
                "./build/crofton_optimized",
                "../build_optimized/crofton_optimized",
                "./crofton_optimized"
            ]

            for path in possible_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    binary_path = path
                    break

            if binary_path is None:
                raise FileNotFoundError(
                    "Metal-optimized binary not found. "
                    "Please build first: cd build_optimized && make crofton_optimized"
                )

        self.binary_path = os.path.abspath(binary_path)

        # Verify binary exists and is executable
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

        if not os.access(self.binary_path, os.X_OK):
            raise PermissionError(f"Binary not executable: {self.binary_path}")

        # Check if Metal is available
        self._check_metal_availability()

        print(f"✅ MetalImageProcessor initialized")
        print(f"📦 Binary: {self.binary_path}")

    def _check_metal_availability(self):
        """Check if Metal is available on the system"""
        try:
            result = subprocess.run(
                [self.binary_path, "--check-metal"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # If binary doesn't support --check-metal, assume Metal is available
            # (actual check happens at runtime)

        except subprocess.TimeoutExpired:
            raise RuntimeError("Metal availability check timed out")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify Metal availability: {e}")

    def process_image(
        self,
        image: np.ndarray,
        params: Optional[Dict] = None,
        return_intermediates: bool = True,
        save_output: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process image using Metal GPU acceleration

        Args:
            image: Input image as numpy array (BGR format)
            params: Processing parameters (HSV range, kernel sizes, etc.)
            return_intermediates: Return intermediate processing stages
            save_output: Save output images to disk
            output_dir: Directory for output images

        Returns:
            Dictionary with results:
            {
                'success': bool,
                'processing_time_ms': float,
                'gpu_time_ms': float,
                'contours': list,
                'descriptor': list,
                'final_image': np.ndarray,
                'intermediates': dict (if return_intermediates=True),
                'metrics': dict,
                'error': str (if success=False)
            }
        """

        start_time = time.time()

        # Create temporary directory for I/O
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save input image
            input_path = temp_dir_path / "input.png"
            cv2.imwrite(str(input_path), image)

            # Prepare output paths
            output_path = temp_dir_path / "output.json"
            result_image_path = temp_dir_path / "result.png"

            # Build command
            cmd = [
                self.binary_path,
                str(input_path),
                "--json-output", str(output_path)
            ]

            if return_intermediates:
                cmd.extend(["--save-intermediates", str(temp_dir_path)])

            if params:
                # Convert params to JSON and pass as argument
                params_path = temp_dir_path / "params.json"
                with open(params_path, 'w') as f:
                    json.dump(params, f)
                cmd.extend(["--params", str(params_path)])

            try:
                # Execute Metal processor
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )

                if result.returncode != 0:
                    return {
                        'success': False,
                        'error': f"Metal processor failed: {result.stderr}",
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }

                # Parse JSON output
                if output_path.exists():
                    with open(output_path, 'r') as f:
                        output_data = json.load(f)
                else:
                    # Fallback: parse stdout for results
                    output_data = self._parse_stdout(result.stdout)

                # Load result image if available
                final_image = None
                if result_image_path.exists():
                    final_image = cv2.imread(str(result_image_path))

                # Load intermediate images if requested
                intermediates = {}
                if return_intermediates:
                    intermediate_files = {
                        'hsv_mask': 'hsv_mask.png',
                        'tophat': 'tophat.png',
                        'tophat_binary': 'tophat_binary.png',
                        'combined': 'combined.png',
                        'opened': 'opened.png',
                        'closed': 'closed.png'
                    }

                    for key, filename in intermediate_files.items():
                        filepath = temp_dir_path / filename
                        if filepath.exists():
                            intermediates[key] = cv2.imread(str(filepath))

                # Save outputs if requested
                if save_output and output_dir:
                    self._save_outputs(
                        output_dir,
                        final_image,
                        intermediates,
                        output_data
                    )

                processing_time = (time.time() - start_time) * 1000  # Convert to ms

                return {
                    'success': True,
                    'processing_time_ms': processing_time,
                    'gpu_time_ms': output_data.get('gpu_time_ms', 0),
                    'final_image': final_image,
                    'intermediates': intermediates,
                    'metrics': output_data.get('metrics', {}),
                    'descriptor': output_data.get('descriptor', []),
                    'contours': output_data.get('contours', []),
                    'raw_output': output_data
                }

            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': 'Processing timeout (>30s)'
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Unexpected error: {str(e)}'
                }

    def _parse_stdout(self, stdout: str) -> Dict:
        """Parse stdout for metrics if JSON output not available"""
        metrics = {}

        # Extract processing time
        for line in stdout.split('\n'):
            if 'Total processing time:' in line:
                try:
                    time_ms = float(line.split(':')[1].strip().split()[0])
                    metrics['total_time_ms'] = time_ms
                except:
                    pass

            elif 'GPU time:' in line:
                try:
                    time_ms = float(line.split(':')[1].strip().split()[0])
                    metrics['gpu_time_ms'] = time_ms
                except:
                    pass

        return metrics

    def _save_outputs(
        self,
        output_dir: str,
        final_image: Optional[np.ndarray],
        intermediates: Dict,
        data: Dict
    ):
        """Save outputs to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save final image
        if final_image is not None:
            cv2.imwrite(str(output_path / "result.png"), final_image)

        # Save intermediates
        for key, img in intermediates.items():
            cv2.imwrite(str(output_path / f"{key}.png"), img)

        # Save metrics as JSON
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(data, f, indent=2)

    def benchmark(
        self,
        images: List[np.ndarray],
        num_iterations: int = 5
    ) -> Dict:
        """
        Benchmark Metal processor on multiple images

        Args:
            images: List of input images
            num_iterations: Number of iterations per image

        Returns:
            Benchmark results with timing statistics
        """
        results = {
            'num_images': len(images),
            'num_iterations': num_iterations,
            'timings': [],
            'gpu_timings': [],
            'mean_time_ms': 0,
            'std_time_ms': 0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0,
            'throughput_fps': 0
        }

        print(f"🔬 Benchmarking Metal processor...")
        print(f"   Images: {len(images)}, Iterations: {num_iterations}")

        for i, image in enumerate(images):
            for iteration in range(num_iterations):
                result = self.process_image(
                    image,
                    return_intermediates=False
                )

                if result['success']:
                    time_ms = result['processing_time_ms']
                    gpu_time_ms = result.get('gpu_time_ms', 0)

                    results['timings'].append(time_ms)
                    results['gpu_timings'].append(gpu_time_ms)

                    print(f"   Image {i+1}/{len(images)}, "
                          f"Iter {iteration+1}/{num_iterations}: "
                          f"{time_ms:.1f}ms (GPU: {gpu_time_ms:.1f}ms)")

        # Calculate statistics
        if results['timings']:
            results['mean_time_ms'] = np.mean(results['timings'])
            results['std_time_ms'] = np.std(results['timings'])
            results['min_time_ms'] = np.min(results['timings'])
            results['max_time_ms'] = np.max(results['timings'])
            results['throughput_fps'] = 1000.0 / results['mean_time_ms']
            results['gpu_mean_time_ms'] = np.mean(results['gpu_timings'])

        print(f"\n✅ Benchmark complete:")
        print(f"   Mean: {results['mean_time_ms']:.1f} ± {results['std_time_ms']:.1f} ms")
        print(f"   Range: [{results['min_time_ms']:.1f}, {results['max_time_ms']:.1f}] ms")
        print(f"   Throughput: {results['throughput_fps']:.1f} fps")

        return results


def main():
    """Example usage and testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Metal Image Processor Python Wrapper')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    args = parser.parse_args()

    # Initialize processor
    processor = MetalImageProcessor()

    if args.benchmark:
        # Load test image
        if args.image:
            image = cv2.imread(args.image)
            if image is None:
                print(f"❌ Could not load image: {args.image}")
                return

            # Run benchmark
            results = processor.benchmark([image], num_iterations=10)

            # Print results
            print("\n" + "="*60)
            print("BENCHMARK RESULTS")
            print("="*60)
            print(f"Mean processing time: {results['mean_time_ms']:.2f} ms")
            print(f"Std deviation: {results['std_time_ms']:.2f} ms")
            print(f"Throughput: {results['throughput_fps']:.2f} fps")
            print("="*60)

    elif args.image:
        # Process single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ Could not load image: {args.image}")
            return

        result = processor.process_image(
            image,
            return_intermediates=True,
            save_output=args.output_dir is not None,
            output_dir=args.output_dir
        )

        if result['success']:
            print("✅ Processing successful!")
            print(f"⏱️  Processing time: {result['processing_time_ms']:.1f} ms")
            print(f"⚡ GPU time: {result['gpu_time_ms']:.1f} ms")

            if args.output_dir:
                print(f"💾 Results saved to: {args.output_dir}")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
