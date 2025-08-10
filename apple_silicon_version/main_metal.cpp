#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Configuration constants
static const int CROFTON_MAX_POINTS = 239;
static const float PI = 3.1415927f;

// Declare Metal Crofton descriptor function (implemented in crofton_metal.mm)
extern "C" void computeMetalCroftonDescriptor(const std::vector<float>& contourData, 
                                              std::vector<float>& descriptor);

// Enhanced preprocessing function - same as our corrected CUDA version
Mat enhancedPreprocessing(const Mat& inputImage) {
    cout << "Applying CUDA-replicated preprocessing (HSV + Top-Hat + Morphology)..." << endl;
    
    Mat hsv;
    cvtColor(inputImage, hsv, COLOR_BGR2HSV);
    
    // Exact same HSV range as CUDA: [100,20,20] to [180,255,255]
    Scalar lowerBound(100, 20, 20);
    Scalar upperBound(180, 255, 255);
    Mat maskHSV;
    inRange(hsv, lowerBound, upperBound, maskHSV);
    cout << "HSV mask created, white pixels: " << countNonZero(maskHSV) << endl;
    
    // Top-Hat transform on grayscale - exact same as CUDA
    Mat imgGray;
    cvtColor(inputImage, imgGray, COLOR_BGR2GRAY);
    
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15)); // Same 15x15 kernel
    Mat opened, topHat;
    morphologyEx(imgGray, opened, MORPH_OPEN, kernel);
    topHat = imgGray - opened;
    
    // Binarize with Otsu - same as CUDA
    Mat binTopHat;
    threshold(topHat, binTopHat, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cout << "Top-Hat binarized, white pixels: " << countNonZero(binTopHat) << endl;
    
    // Combine with logical OR - same as CUDA
    Mat combined;
    bitwise_or(maskHSV, binTopHat, combined);
    cout << "Combined HSV+TopHat, white pixels: " << countNonZero(combined) << endl;
    
    // Two-step morphology: open then close - same as CUDA
    Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat openedResult, closedResult;
    
    morphologyEx(combined, openedResult, MORPH_OPEN, kernelOpen);
    morphologyEx(openedResult, closedResult, MORPH_CLOSE, kernelClose);
    
    cout << "Final morphology - After open: " << countNonZero(openedResult) 
         << ", after close: " << countNonZero(closedResult) << endl;
    
    return closedResult;
}

// Resample contour to fixed number of points
vector<Point> resampleContour(const vector<Point>& contour, int nPoints) {
    vector<Point> resampled;
    if (contour.empty() || nPoints <= 0) return resampled;

    // Compute perimeter and cumulative distances
    vector<double> cumDist;
    cumDist.push_back(0.0);
    double totalLen = 0.0;
    for (size_t i = 0; i < contour.size(); ++i) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()];
        double dx = double(p2.x - p1.x);
        double dy = double(p2.y - p1.y);
        double segLen = sqrt(dx*dx + dy*dy);
        totalLen += segLen;
        cumDist.push_back(totalLen);
    }

    if (totalLen < 1e-9) {
        resampled.resize(nPoints, contour[0]);
        return resampled;
    }

    double step = totalLen / nPoints;
    resampled.reserve(nPoints);

    for (int i = 0; i < nPoints; ++i) {
        double currentDist = i * step;
        while (currentDist >= totalLen) currentDist -= totalLen;

        auto it = std::lower_bound(cumDist.begin(), cumDist.end(), currentDist);
        int idx = int(it - cumDist.begin());
        if (idx == 0) {
            resampled.push_back(contour[0]);
        } else {
            int prevIdx = idx - 1;
            double prevDist = cumDist[prevIdx];
            double segLen = (cumDist[idx] - prevDist);
            double frac = (currentDist - prevDist) / segLen;

            Point p1 = contour[prevIdx];
            Point p2 = contour[idx % contour.size()];

            float x = float(p1.x + frac * (p2.x - p1.x));
            float y = float(p1.y + frac * (p2.y - p1.y));
            resampled.push_back(Point(cvRound(x), cvRound(y)));
        }
    }
    return resampled;
}

int main() {
    cout << "Apple Silicon Metal-Accelerated Edge Detection + Crofton Descriptor" << endl;
    cout << "=====================================================================" << endl;
    cout << "GPU-Accelerated version with Metal compute shaders" << endl;
    cout << endl;
    
    // Load test image
    Mat imgColor = imread("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/test_cell.jpg");
    if (imgColor.empty()) {
        cerr << "Error: Could not load test image" << endl;
        return -1;
    }
    
    cout << "✅ Image loaded: " << imgColor.cols << "x" << imgColor.rows << " pixels" << endl;
    
    // Start timing
    auto startTime = high_resolution_clock::now();
    
    // Show original image
    imshow("Original Image", imgColor);
    
    // Apply corrected CUDA-replicated preprocessing
    cout << "\n🔧 Phase 1: Enhanced preprocessing (CUDA algorithm replication)..." << endl;
    Mat preprocessed = enhancedPreprocessing(imgColor);
    imshow("Enhanced Preprocessing", preprocessed);
    
    // Find contours
    cout << "\n🔍 Phase 2: Finding contours..." << endl;
    vector<vector<Point>> contours;
    findContours(preprocessed, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    if (contours.empty()) {
        cerr << "❌ No contours found" << endl;
        waitKey(0);
        return -1;
    }
    
    cout << "✅ Found " << contours.size() << " contours" << endl;
    
    // Find largest contour - same as CUDA
    int largestIdx = 0;
    double largestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIdx = (int)i;
        }
    }
    
    cout << "✅ Largest contour: area = " << largestArea << " pixels, perimeter = " 
         << cv::arcLength(contours[largestIdx], true) << endl;
    
    // Log average HSV in largest contour (same as CUDA logging)
    Mat hsv;
    cvtColor(imgColor, hsv, COLOR_BGR2HSV);
    int total_h = 0, total_s = 0, total_v = 0, count_hsv = 0;
    
    for (const Point& pt : contours[largestIdx]) {
        int x = pt.x, y = pt.y;
        if (x >= 0 && x < hsv.cols && y >= 0 && y < hsv.rows) {
            Vec3b hsv_pixel = hsv.at<Vec3b>(y, x);
            total_h += hsv_pixel[0];
            total_s += hsv_pixel[1];
            total_v += hsv_pixel[2];
            count_hsv++;
        }
    }
    
    if (count_hsv > 0) {
        float avg_h = float(total_h) / count_hsv;
        float avg_s = float(total_s) / count_hsv;
        float avg_v = float(total_v) / count_hsv;
        cout << "✅ Average H,S,V in largest contour: " << avg_h << ", " << avg_s << ", " << avg_v << endl;
    }
    
    // Visualize largest contour
    Mat result = imgColor.clone();
    drawContours(result, contours, largestIdx, Scalar(0, 255, 0), 3);
    imshow("Detected Cell Boundary", result);
    
    // Resample and center contour for Crofton descriptor
    cout << "\n⚙️  Phase 3: Preparing contour for Crofton descriptor..." << endl;
    vector<Point> resampled = resampleContour(contours[largestIdx], CROFTON_MAX_POINTS);
    cout << "✅ Resampled to " << resampled.size() << " points" << endl;
    
    // Center the contour
    Rect box = boundingRect(resampled);
    float cx = box.x + box.width * 0.5f;
    float cy = box.y + box.height * 0.5f;
    for (auto& pt : resampled) {
        pt.x = cvRound(pt.x - cx);
        pt.y = cvRound(pt.y - cy);
    }
    cout << "✅ Contour centered at origin" << endl;
    
    // Convert to contour data format for Metal GPU processing
    vector<float> contourData;
    contourData.reserve(resampled.size() * 2);
    for (const Point& p : resampled) {
        contourData.push_back((float)p.x);
    }
    for (const Point& p : resampled) {
        contourData.push_back((float)p.y);
    }
    
    cout << "✅ Contour data prepared: " << contourData.size() << " floats" << endl;
    
    // 🚀 Metal GPU-Accelerated Crofton Descriptor Computation
    cout << "\n🚀 Phase 4: Metal GPU-accelerated Crofton descriptor computation..." << endl;
    vector<float> descriptor;
    
    try {
        auto metalStartTime = high_resolution_clock::now();
        computeMetalCroftonDescriptor(contourData, descriptor);
        auto metalEndTime = high_resolution_clock::now();
        
        auto metalDuration = duration_cast<milliseconds>(metalEndTime - metalStartTime);
        cout << "⚡ Metal GPU computation time: " << metalDuration.count() << " ms" << endl;
        
    } catch (const exception& e) {
        cerr << "❌ Metal computation failed: " << e.what() << endl;
        cerr << "💡 Falling back to CPU computation..." << endl;
        
        // Could implement CPU fallback here if needed
        cout << "❌ No CPU fallback implemented in this version" << endl;
        return -1;
    }
    
    auto endTime = high_resolution_clock::now();
    auto totalDuration = duration_cast<milliseconds>(endTime - startTime);
    
    // Output results
    cout << "\n🎉 Results:" << endl;
    cout << "==========" << endl;
    cout << "✅ Total processing time: " << totalDuration.count() << " ms" << endl;
    cout << "✅ Crofton descriptor computed with " << descriptor.size() << " angles" << endl;
    cout << "✅ Sample descriptor values: ";
    for (int i = 0; i < min(10, (int)descriptor.size()); ++i) {
        cout << descriptor[i] << " ";
    }
    cout << endl;
    
    // Save results to file
    cout << "\n💾 Saving results..." << endl;
    ofstream outFile("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/metal_crofton_result.txt");
    if (outFile.is_open()) {
        outFile << "Apple Silicon Metal-Accelerated Crofton Descriptor Results\n";
        outFile << "========================================================\n";
        outFile << "Processing time: " << totalDuration.count() << " ms\n";
        outFile << "Image size: " << imgColor.cols << "x" << imgColor.rows << "\n";
        outFile << "Contours found: " << contours.size() << "\n";
        outFile << "Largest contour area: " << largestArea << " pixels\n";
        outFile << "Boundary points: " << CROFTON_MAX_POINTS << "\n";
        outFile << "Projection angles: " << descriptor.size() << "\n\n";
        
        outFile << "Algorithm Pipeline:\n";
        outFile << "1. HSV color masking [100,20,20] to [180,255,255] (CUDA-replicated)\n";
        outFile << "2. Top-hat morphological transform (15x15 ellipse)\n";
        outFile << "3. Logical OR combination\n";
        outFile << "4. Two-step morphology: open(3x3) + close(5x5)\n";
        outFile << "5. Largest contour selection\n";
        outFile << "6. 🚀 Metal GPU-accelerated Crofton descriptor\n\n";
        
        outFile << "Metal GPU Enhancements:\n";
        outFile << "- Native Apple Silicon Metal compute shaders\n";
        outFile << "- Parallel projection computation (GPU)\n";
        outFile << "- Parallel descriptor computation (1 thread per angle)\n";
        outFile << "- Enhanced BANDA parameter (20.0 vs 10.0)\n";
        outFile << "- Gradual attenuation instead of hard cutoff\n\n";
        
        outFile << "Descriptor values:\n";
        for (size_t i = 0; i < descriptor.size(); ++i) {
            outFile << "Angle " << i << ": " << descriptor[i] << "\n";
        }
        outFile.close();
        cout << "Results saved to metal_crofton_result.txt" << endl;
    }
    
    cout << "\nMetal-accelerated edge detection completed successfully!" << endl;
    cout << "Performance: Native Apple Silicon GPU acceleration" << endl;
    cout << "Algorithm: CUDA-replicated edge detection + Metal Crofton descriptor" << endl;
    cout << "\nPress any key to exit..." << endl;
    
    waitKey(0);
    return 0;
}