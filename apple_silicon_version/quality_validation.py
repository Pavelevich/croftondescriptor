#!/usr/bin/env python3
"""
Quality Validation System
Compares Metal GPU results vs CPU baseline for numerical equivalence
"""

import cv2
import numpy as np
import subprocess
import json
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class ValidationMetrics:
    """Metrics for quality validation"""
    iou: float  # Intersection over Union
    hausdorff_distance: float  # Hausdorff distance in pixels
    descriptor_correlation: float  # Pearson correlation of descriptors
    contour_similarity: float  # Contour similarity (0-1)
    pixel_accuracy: float  # Pixel-wise accuracy
    passed: bool  # Overall pass/fail

    def __str__(self):
        return f"""
Quality Validation Metrics:
  IoU (Intersection over Union):    {self.iou:.4f}  {'✅' if self.iou > 0.95 else '❌'}
  Hausdorff Distance:                {self.hausdorff_distance:.2f}px  {'✅' if self.hausdorff_distance < 5 else '❌'}
  Descriptor Correlation:            {self.descriptor_correlation:.4f}  {'✅' if self.descriptor_correlation > 0.98 else '❌'}
  Contour Similarity:                {self.contour_similarity:.4f}  {'✅' if self.contour_similarity > 0.95 else '❌'}
  Pixel Accuracy:                    {self.pixel_accuracy:.4f}  {'✅' if self.pixel_accuracy > 0.98 else '❌'}
  Overall Status:                    {'✅ PASSED' if self.passed else '❌ FAILED'}
"""

class QualityValidator:
    """Validates Metal implementation against CPU baseline"""

    def __init__(self, build_dir: str = "./build_optimized"):
        self.build_dir = Path(build_dir)
        self.cpu_binary = self.build_dir / "crofton_simple"
        self.metal_binary = self.build_dir / "crofton_optimized"

        # Validation thresholds
        self.thresholds = {
            'iou': 0.95,
            'hausdorff': 5.0,  # pixels
            'descriptor_correlation': 0.98,
            'contour_similarity': 0.95,
            'pixel_accuracy': 0.98
        }

    def compute_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """Compute Intersection over Union"""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)

        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
        return iou

    def compute_hausdorff_distance(
        self,
        contour1: np.ndarray,
        contour2: np.ndarray
    ) -> float:
        """Compute Hausdorff distance between two contours"""
        if len(contour1) == 0 or len(contour2) == 0:
            return float('inf')

        # Forward Hausdorff (contour1 to contour2)
        dist_forward = np.max([
            np.min([np.linalg.norm(p1 - p2) for p2 in contour2])
            for p1 in contour1
        ])

        # Backward Hausdorff (contour2 to contour1)
        dist_backward = np.max([
            np.min([np.linalg.norm(p2 - p1) for p1 in contour1])
            for p2 in contour2
        ])

        return max(dist_forward, dist_backward)

    def compute_descriptor_correlation(
        self,
        desc1: List[float],
        desc2: List[float]
    ) -> float:
        """Compute Pearson correlation between descriptors"""
        if len(desc1) != len(desc2):
            return 0.0

        correlation = np.corrcoef(desc1, desc2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def extract_contour_from_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
        """Extract largest contour from binary mask"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return np.array([])

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        return largest.squeeze()

    def run_processor(
        self,
        binary_path: Path,
        image_path: Path,
        temp_dir: Path
    ) -> Dict:
        """Run processor and extract results"""

        try:
            result = subprocess.run(
                [str(binary_path), str(image_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(temp_dir)
            )

            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}

            # Find output files
            output_files = list(temp_dir.glob("*.png")) + list(temp_dir.glob("*.txt"))

            # Load final mask (if available)
            final_mask = None
            for f in temp_dir.glob("*final*.png"):
                final_mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                break

            if final_mask is None:
                for f in temp_dir.glob("*closed*.png"):
                    final_mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    break

            # Parse descriptor from output
            descriptor = []
            for f in temp_dir.glob("*.txt"):
                with open(f, 'r') as file:
                    for line in file:
                        if line.startswith("Angle"):
                            try:
                                value = float(line.split(':')[1].strip())
                                descriptor.append(value)
                            except:
                                pass

            return {
                'success': True,
                'mask': final_mask,
                'descriptor': descriptor
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def validate_image(
        self,
        image_path: Path,
        verbose: bool = True
    ) -> ValidationMetrics:
        """Validate Metal implementation against CPU for single image"""

        if verbose:
            print(f"\n🔍 Validating: {image_path.name}")

        import tempfile
        import shutil

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_cpu, \
             tempfile.TemporaryDirectory() as temp_metal:

            temp_cpu_path = Path(temp_cpu)
            temp_metal_path = Path(temp_metal)

            # Copy image to temp directories
            cpu_image = temp_cpu_path / image_path.name
            metal_image = temp_metal_path / image_path.name
            shutil.copy(image_path, cpu_image)
            shutil.copy(image_path, metal_image)

            # Run CPU baseline
            if verbose:
                print("   🖥️  Running CPU baseline...")

            cpu_result = self.run_processor(
                self.cpu_binary,
                cpu_image,
                temp_cpu_path
            )

            if not cpu_result['success']:
                print(f"   ❌ CPU processing failed: {cpu_result.get('error', 'Unknown')}")
                return ValidationMetrics(0, float('inf'), 0, 0, 0, False)

            # Run Metal GPU
            if verbose:
                print("   ⚡ Running Metal GPU...")

            metal_result = self.run_processor(
                self.metal_binary,
                metal_image,
                temp_metal_path
            )

            if not metal_result['success']:
                print(f"   ❌ Metal processing failed: {metal_result.get('error', 'Unknown')}")
                return ValidationMetrics(0, float('inf'), 0, 0, 0, False)

            # Compute metrics
            if verbose:
                print("   📊 Computing quality metrics...")

            # 1. IoU
            iou = 0.0
            if cpu_result['mask'] is not None and metal_result['mask'] is not None:
                iou = self.compute_iou(
                    cpu_result['mask'] > 127,
                    metal_result['mask'] > 127
                )

            # 2. Hausdorff distance
            hausdorff = float('inf')
            if cpu_result['mask'] is not None and metal_result['mask'] is not None:
                cpu_contour = self.extract_contour_from_mask(cpu_result['mask'])
                metal_contour = self.extract_contour_from_mask(metal_result['mask'])

                if len(cpu_contour) > 0 and len(metal_contour) > 0:
                    hausdorff = self.compute_hausdorff_distance(
                        cpu_contour,
                        metal_contour
                    )

            # 3. Descriptor correlation
            descriptor_corr = 0.0
            if cpu_result['descriptor'] and metal_result['descriptor']:
                descriptor_corr = self.compute_descriptor_correlation(
                    cpu_result['descriptor'],
                    metal_result['descriptor']
                )

            # 4. Contour similarity (shape matching)
            contour_sim = 0.0
            if cpu_result['mask'] is not None and metal_result['mask'] is not None:
                cpu_contour = self.extract_contour_from_mask(cpu_result['mask'])
                metal_contour = self.extract_contour_from_mask(metal_result['mask'])

                if len(cpu_contour) > 0 and len(metal_contour) > 0:
                    match_val = cv2.matchShapes(
                        cpu_contour,
                        metal_contour,
                        cv2.CONTOURS_MATCH_I1,
                        0.0
                    )
                    contour_sim = 1.0 / (1.0 + match_val)  # Convert to similarity

            # 5. Pixel accuracy
            pixel_acc = 0.0
            if cpu_result['mask'] is not None and metal_result['mask'] is not None:
                agreement = np.sum(
                    (cpu_result['mask'] > 127) == (metal_result['mask'] > 127)
                )
                total = cpu_result['mask'].size
                pixel_acc = agreement / total

            # Determine pass/fail
            passed = (
                iou >= self.thresholds['iou'] and
                hausdorff <= self.thresholds['hausdorff'] and
                descriptor_corr >= self.thresholds['descriptor_correlation'] and
                contour_sim >= self.thresholds['contour_similarity'] and
                pixel_acc >= self.thresholds['pixel_accuracy']
            )

            metrics = ValidationMetrics(
                iou=iou,
                hausdorff_distance=hausdorff,
                descriptor_correlation=descriptor_corr,
                contour_similarity=contour_sim,
                pixel_accuracy=pixel_acc,
                passed=passed
            )

            if verbose:
                print(metrics)

            return metrics

    def validate_dataset(
        self,
        image_paths: List[Path],
        output_dir: Path = Path("./validation_results")
    ) -> Dict:
        """Validate on entire dataset"""

        output_dir.mkdir(parents=True, exist_ok=True)

        print("╔════════════════════════════════════════════════════════════╗")
        print("║          Quality Validation Suite                         ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"\n📊 Validating {len(image_paths)} images...")

        all_metrics = []

        for img_path in image_paths:
            metrics = self.validate_image(img_path, verbose=True)
            all_metrics.append(metrics)

        # Aggregate results
        passed_count = sum(1 for m in all_metrics if m.passed)
        total_count = len(all_metrics)

        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total images:     {total_count}")
        print(f"Passed:           {passed_count} ({100*passed_count/total_count:.1f}%)")
        print(f"Failed:           {total_count - passed_count}")
        print()

        # Average metrics
        avg_iou = np.mean([m.iou for m in all_metrics])
        avg_hausdorff = np.mean([m.hausdorff_distance for m in all_metrics if m.hausdorff_distance != float('inf')])
        avg_desc_corr = np.mean([m.descriptor_correlation for m in all_metrics])
        avg_contour_sim = np.mean([m.contour_similarity for m in all_metrics])
        avg_pixel_acc = np.mean([m.pixel_accuracy for m in all_metrics])

        print("Average Metrics:")
        print(f"  IoU:                    {avg_iou:.4f}")
        print(f"  Hausdorff Distance:     {avg_hausdorff:.2f}px")
        print(f"  Descriptor Correlation: {avg_desc_corr:.4f}")
        print(f"  Contour Similarity:     {avg_contour_sim:.4f}")
        print(f"  Pixel Accuracy:         {avg_pixel_acc:.4f}")

        # Save results
        results = {
            'total': total_count,
            'passed': passed_count,
            'failed': total_count - passed_count,
            'average_metrics': {
                'iou': avg_iou,
                'hausdorff_distance': avg_hausdorff,
                'descriptor_correlation': avg_desc_corr,
                'contour_similarity': avg_contour_sim,
                'pixel_accuracy': avg_pixel_acc
            },
            'thresholds': self.thresholds,
            'individual_results': [
                {
                    'image': str(img_path.name),
                    'metrics': {
                        'iou': m.iou,
                        'hausdorff': m.hausdorff_distance,
                        'descriptor_corr': m.descriptor_correlation,
                        'contour_sim': m.contour_similarity,
                        'pixel_acc': m.pixel_accuracy
                    },
                    'passed': m.passed
                }
                for img_path, m in zip(image_paths, all_metrics)
            ]
        }

        json_path = output_dir / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {json_path}")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Quality Validation for Metal Optimization')
    parser.add_argument('images', nargs='+', help='Test image paths')
    parser.add_argument('--build-dir', default='./build_optimized', help='Build directory')
    parser.add_argument('--output-dir', default='./validation_results', help='Output directory')

    args = parser.parse_args()

    image_paths = [Path(img) for img in args.images]

    validator = QualityValidator(build_dir=args.build_dir)
    results = validator.validate_dataset(image_paths, output_dir=Path(args.output_dir))

    if results['passed'] == results['total']:
        print("\n✅ All validations passed!")
        return 0
    else:
        print(f"\n⚠️  {results['failed']} validation(s) failed")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
