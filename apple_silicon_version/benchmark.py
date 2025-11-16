#!/usr/bin/env python3
"""
Automated Benchmarking System for Metal Optimization
Compares CPU baseline vs Metal GPU acceleration
"""

import cv2
import numpy as np
import subprocess
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
import sys

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    implementation: str  # 'cpu', 'metal', 'metal_optimized'
    image_name: str
    image_size: Tuple[int, int]  # (width, height)
    processing_time_ms: float
    gpu_time_ms: float
    memory_mb: float
    success: bool
    error_message: str = ""

class BenchmarkSuite:
    """Comprehensive benchmarking suite for Metal optimization"""

    def __init__(self, build_dir: str = "./build_optimized"):
        self.build_dir = Path(build_dir)
        self.results: List[BenchmarkResult] = []

        # Check binaries exist
        self.binaries = {
            'cpu': self.build_dir / "crofton_simple",
            'metal': self.build_dir / "crofton_metal",
            'metal_optimized': self.build_dir / "crofton_optimized"
        }

        for name, path in self.binaries.items():
            if not path.exists():
                print(f"⚠️  Warning: {name} binary not found at {path}")

    def run_single_benchmark(
        self,
        binary_path: Path,
        image_path: Path,
        implementation: str
    ) -> BenchmarkResult:
        """Run benchmark on single image with specific binary"""

        image = cv2.imread(str(image_path))
        if image is None:
            return BenchmarkResult(
                implementation=implementation,
                image_name=image_path.name,
                image_size=(0, 0),
                processing_time_ms=0,
                gpu_time_ms=0,
                memory_mb=0,
                success=False,
                error_message=f"Could not load image: {image_path}"
            )

        image_size = (image.shape[1], image.shape[0])

        try:
            start_time = time.time()

            result = subprocess.run(
                [str(binary_path), str(image_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            end_time = time.time()
            wall_time_ms = (end_time - start_time) * 1000

            # Parse output for GPU time
            gpu_time_ms = 0.0
            for line in result.stdout.split('\n'):
                if 'GPU time:' in line or 'Metal' in line:
                    try:
                        # Extract time value
                        parts = line.split(':')
                        if len(parts) >= 2:
                            time_str = parts[1].strip().split()[0]
                            gpu_time_ms = float(time_str)
                    except:
                        pass

            success = result.returncode == 0

            return BenchmarkResult(
                implementation=implementation,
                image_name=image_path.name,
                image_size=image_size,
                processing_time_ms=wall_time_ms,
                gpu_time_ms=gpu_time_ms,
                memory_mb=0,  # TODO: Measure actual memory usage
                success=success,
                error_message="" if success else result.stderr[:200]
            )

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                implementation=implementation,
                image_name=image_path.name,
                image_size=image_size,
                processing_time_ms=60000,
                gpu_time_ms=0,
                memory_mb=0,
                success=False,
                error_message="Timeout (>60s)"
            )
        except Exception as e:
            return BenchmarkResult(
                implementation=implementation,
                image_name=image_path.name,
                image_size=image_size,
                processing_time_ms=0,
                gpu_time_ms=0,
                memory_mb=0,
                success=False,
                error_message=str(e)
            )

    def run_full_benchmark(
        self,
        test_images: List[Path],
        num_iterations: int = 5
    ) -> pd.DataFrame:
        """Run comprehensive benchmark on all images and implementations"""

        print("╔════════════════════════════════════════════════════════════╗")
        print("║         Metal Optimization Benchmark Suite                ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"\n📊 Configuration:")
        print(f"   Test images: {len(test_images)}")
        print(f"   Iterations per image: {num_iterations}")
        print(f"   Total runs: {len(test_images) * len(self.binaries) * num_iterations}")
        print()

        self.results = []

        for image_path in test_images:
            print(f"\n🖼️  Testing: {image_path.name}")

            for impl_name, binary_path in self.binaries.items():
                if not binary_path.exists():
                    print(f"   ⏭️  Skipping {impl_name} (binary not found)")
                    continue

                print(f"   🔄 {impl_name}:", end=" ", flush=True)

                iteration_times = []

                for i in range(num_iterations):
                    result = self.run_single_benchmark(
                        binary_path,
                        image_path,
                        impl_name
                    )

                    self.results.append(result)

                    if result.success:
                        iteration_times.append(result.processing_time_ms)
                        print("✓", end="", flush=True)
                    else:
                        print("✗", end="", flush=True)

                if iteration_times:
                    mean_time = np.mean(iteration_times)
                    std_time = np.std(iteration_times)
                    print(f" → {mean_time:.1f} ± {std_time:.1f} ms")
                else:
                    print(" → FAILED")

        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        return df

    def generate_report(
        self,
        df: pd.DataFrame,
        output_dir: Path = Path("./benchmark_results")
    ):
        """Generate comprehensive benchmark report"""

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)

        # Calculate aggregate statistics
        successful = df[df['success'] == True]

        if successful.empty:
            print("❌ No successful runs")
            return

        stats = successful.groupby('implementation')['processing_time_ms'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).round(2)

        print("\n📊 Processing Time (ms):")
        print(stats)

        # Calculate speedups
        if 'cpu' in stats.index:
            cpu_mean = stats.loc['cpu', 'mean']
            print("\n⚡ Speedup vs CPU:")

            for impl in stats.index:
                if impl != 'cpu':
                    impl_mean = stats.loc[impl, 'mean']
                    speedup = cpu_mean / impl_mean
                    print(f"   {impl:20s}: {speedup:.2f}x faster")

        # Save detailed results to CSV
        csv_path = output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")

        # Generate plots
        self._generate_plots(df, output_dir)

        # Save JSON summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_images': len(df['image_name'].unique()),
            'num_iterations': len(df) // (len(df['implementation'].unique()) * len(df['image_name'].unique())),
            'statistics': stats.to_dict(),
            'total_runs': len(df),
            'successful_runs': len(successful),
            'failed_runs': len(df) - len(successful)
        }

        json_path = output_dir / "benchmark_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📄 Summary saved to: {json_path}")

    def _generate_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate visualization plots"""

        successful = df[df['success'] == True]

        if successful.empty:
            return

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Processing time comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Box plot
        sns.boxplot(
            data=successful,
            x='implementation',
            y='processing_time_ms',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Processing Time Distribution')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].set_xlabel('Implementation')

        # Bar plot with error bars
        stats = successful.groupby('implementation')['processing_time_ms'].agg(['mean', 'std'])
        stats.plot(kind='bar', y='mean', yerr='std', ax=axes[0, 1], legend=False)
        axes[0, 1].set_title('Mean Processing Time ± Std Dev')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].set_xlabel('Implementation')

        # Speedup chart (if CPU baseline exists)
        if 'cpu' in successful['implementation'].values:
            cpu_mean = successful[successful['implementation'] == 'cpu']['processing_time_ms'].mean()
            speedups = {}

            for impl in successful['implementation'].unique():
                if impl != 'cpu':
                    impl_mean = successful[successful['implementation'] == impl]['processing_time_ms'].mean()
                    speedups[impl] = cpu_mean / impl_mean

            if speedups:
                pd.Series(speedups).plot(kind='bar', ax=axes[1, 0], color='green')
                axes[1, 0].set_title('Speedup vs CPU Baseline')
                axes[1, 0].set_ylabel('Speedup (x)')
                axes[1, 0].axhline(y=1, color='r', linestyle='--', label='CPU baseline')
                axes[1, 0].legend()

        # Image size vs processing time
        sns.scatterplot(
            data=successful,
            x=successful['image_size'].apply(lambda x: x[0] * x[1] / 1000000),  # Megapixels
            y='processing_time_ms',
            hue='implementation',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Processing Time vs Image Size')
        axes[1, 1].set_xlabel('Image Size (Megapixels)')
        axes[1, 1].set_ylabel('Time (ms)')

        plt.tight_layout()
        plot_path = output_dir / "benchmark_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plots saved to: {plot_path}")

        plt.close()

def main():
    """Main benchmark execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Metal Optimization Benchmark Suite')
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Test image paths'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of iterations per image (default: 5)'
    )
    parser.add_argument(
        '--build-dir',
        type=str,
        default='./build_optimized',
        help='Build directory path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Gather test images
    test_images = []

    if args.images:
        test_images.extend([Path(img) for img in args.images])

    if args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            test_images.extend(image_dir.glob(ext))

    if not test_images:
        print("❌ No test images specified")
        print("   Use --images path1.jpg path2.jpg")
        print("   Or --image-dir /path/to/images/")
        return 1

    test_images = list(set(test_images))  # Remove duplicates
    print(f"✅ Found {len(test_images)} test images")

    # Run benchmark
    suite = BenchmarkSuite(build_dir=args.build_dir)
    df = suite.run_full_benchmark(test_images, num_iterations=args.iterations)

    # Generate report
    suite.generate_report(df, output_dir=Path(args.output_dir))

    print("\n✅ Benchmark complete!")

    return 0

if __name__ == '__main__':
    sys.exit(main())
