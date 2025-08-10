#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

// Configuration constants
static const int CROFTON_MAX_POINTS = 239;
static const float PI = 3.1415927f;

// Simple Sobel edge detection function
Mat enhancedSobelEdgeDetection(const Mat& inputImage) {
    Mat grayImage;
    if (inputImage.channels() == 3) {
        cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
    }
    
    // Apply Gaussian blur to reduce noise
    Mat blurred;
    GaussianBlur(grayImage, blurred, Size(3, 3), 0);
    
    // Multi-scale Sobel edge detection
    Mat sobelX_3x3, sobelY_3x3;
    Mat sobelX_5x5, sobelY_5x5;
    
    // 3x3 Sobel
    Sobel(blurred, sobelX_3x3, CV_32F, 1, 0, 3);
    Sobel(blurred, sobelY_3x3, CV_32F, 0, 1, 3);
    
    // 5x5 Sobel
    Sobel(blurred, sobelX_5x5, CV_32F, 1, 0, 5);
    Sobel(blurred, sobelY_5x5, CV_32F, 0, 1, 5);
    
    // Combine gradients with different weights
    Mat magnitude;
    magnitude = 0.7 * (abs(sobelX_3x3) + abs(sobelY_3x3)) + 
                0.3 * (abs(sobelX_5x5) + abs(sobelY_5x5));
    
    // Apply threshold
    Mat binaryResult;
    threshold(magnitude, binaryResult, 30, 255, THRESH_BINARY);
    binaryResult.convertTo(binaryResult, CV_8U);
    
    return binaryResult;
}

// Resample contour function
vector<Point> resampleContour(const vector<Point>& contour, int nPoints) {
    vector<Point> resampled;
    if (contour.empty() || nPoints <= 0) return resampled;

    // Compute perimeter
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

// Enhanced preprocessing function
Mat enhancedPreprocessing(const Mat& inputImage) {
    Mat hsv;
    cvtColor(inputImage, hsv, COLOR_BGR2HSV);
    
    // Expanded HSV range for better color detection
    Scalar lowerBound(0, 30, 30);     // More inclusive range
    Scalar upperBound(180, 255, 255);
    Mat maskHSV;
    inRange(hsv, lowerBound, upperBound, maskHSV);
    
    // Multi-scale top-hat transform
    Mat imgGray;
    cvtColor(inputImage, imgGray, COLOR_BGR2GRAY);
    
    // Apply different kernel sizes for multi-scale edge detection
    vector<Mat> topHats;
    vector<int> kernelSizes = {7, 15, 25}; // Multi-scale approach
    
    for (int kernelSize : kernelSizes) {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernelSize, kernelSize));
        Mat opened, topHat;
        morphologyEx(imgGray, opened, MORPH_OPEN, kernel);
        topHat = imgGray - opened;
        topHats.push_back(topHat);
    }
    
    // Combine multi-scale top-hats
    Mat combinedTopHat = Mat::zeros(imgGray.size(), CV_8U);
    for (const Mat& th : topHats) {
        Mat binTH;
        threshold(th, binTH, 0, 255, THRESH_BINARY | THRESH_OTSU);
        combinedTopHat = max(combinedTopHat, binTH);
    }
    
    // Combine with HSV mask
    Mat combined;
    bitwise_or(maskHSV, combinedTopHat, combined);
    
    // Enhanced morphological operations
    Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(2,2));  // Smaller kernel
    Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(7,7)); // Larger closing
    Mat opened, closed;
    
    morphologyEx(combined, opened, MORPH_OPEN, kernelOpen);
    morphologyEx(opened, closed, MORPH_CLOSE, kernelClose);
    
    return closed;
}

// Simple Crofton descriptor computation
vector<float> computeCroftonDescriptor(const vector<float>& contourData) {
    int numPoints = contourData.size() / 2;
    int numAngles = 361;
    vector<float> descriptor(numAngles);
    
    for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
        float angle = (angleIdx * PI) / 180.0f;
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        // Project all points onto the line perpendicular to angle
        float minProj = INFINITY;
        float maxProj = -INFINITY;
        
        for (int i = 0; i < numPoints; ++i) {
            float x = contourData[i];
            float y = contourData[numPoints + i];
            float proj = x * cosA + y * sinA;
            
            minProj = min(minProj, proj);
            maxProj = max(maxProj, proj);
        }
        
        // Store the width for this angle
        descriptor[angleIdx] = maxProj - minProj;
    }
    
    return descriptor;
}

int main() {
    cout << "Apple Silicon Enhanced Edge Detection - Simple Version" << endl;
    cout << "=====================================================" << endl;
    
    // Load image
    Mat imgColor = imread("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/test_cell.jpg");
    if (imgColor.empty()) {
        cerr << "Error: Could not load image" << endl;
        return -1;
    }
    
    cout << "Image loaded successfully: " << imgColor.cols << "x" << imgColor.rows << endl;
    
    // Show original image
    imshow("Original Image", imgColor);
    
    // Enhanced preprocessing
    cout << "Applying enhanced preprocessing..." << endl;
    Mat preprocessed = enhancedPreprocessing(imgColor);
    imshow("Enhanced Preprocessing", preprocessed);
    
    // Apply enhanced Sobel edge detection
    cout << "Applying multi-scale Sobel edge detection..." << endl;
    Mat edges = enhancedSobelEdgeDetection(preprocessed);
    imshow("Enhanced Edge Detection", edges);
    
    // Find contours
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    if (contours.empty()) {
        cerr << "No contours found" << endl;
        waitKey(0);
        return -1;
    }
    
    cout << "Found " << contours.size() << " contours" << endl;
    
    // Find largest contour
    int largestIdx = 0;
    double largestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIdx = (int)i;
        }
    }
    
    cout << "Largest contour area: " << largestArea << " pixels" << endl;
    
    // Visualize result
    Mat result = imgColor.clone();
    drawContours(result, contours, largestIdx, Scalar(0, 255, 0), 3);
    imshow("Enhanced Edge Detection Result", result);
    
    // Resample and center contour
    cout << "Resampling contour to " << CROFTON_MAX_POINTS << " points..." << endl;
    vector<Point> resampled = resampleContour(contours[largestIdx], CROFTON_MAX_POINTS);
    
    // Center the contour
    Rect box = boundingRect(resampled);
    float cx = box.x + box.width * 0.5f;
    float cy = box.y + box.height * 0.5f;
    for (auto& pt : resampled) {
        pt.x = cvRound(pt.x - cx);
        pt.y = cvRound(pt.y - cy);
    }
    
    // Convert to contour data format
    vector<float> contourData;
    contourData.reserve(resampled.size() * 2);
    for (const Point& p : resampled) {
        contourData.push_back((float)p.x);
    }
    for (const Point& p : resampled) {
        contourData.push_back((float)p.y);
    }
    
    // Compute Crofton descriptor
    cout << "Computing Crofton descriptor..." << endl;
    vector<float> descriptor = computeCroftonDescriptor(contourData);
    
    // Output results
    cout << "\nResults:" << endl;
    cout << "========" << endl;
    cout << "Crofton descriptor computed with " << descriptor.size() << " angles" << endl;
    cout << "Sample descriptor values: ";
    for (int i = 0; i < min(10, (int)descriptor.size()); ++i) {
        cout << descriptor[i] << " ";
    }
    cout << endl;
    
    // Save results to file
    ofstream outFile("/Users/pchmirenko/Desktop/croftondescriptor/apple_silicon_version/simple_crofton_result.txt");
    if (outFile.is_open()) {
        outFile << "Apple Silicon Enhanced Edge Detection - Simple Version Results\n";
        outFile << "============================================================\n";
        outFile << "Number of boundary points: " << CROFTON_MAX_POINTS << "\n";
        outFile << "Number of projection angles: " << descriptor.size() << "\n";
        outFile << "Contour area: " << largestArea << " pixels\n\n";
        outFile << "Improvements made:\n";
        outFile << "- Multi-scale Sobel edge detection (3x3 + 5x5 kernels)\n";
        outFile << "- Expanded HSV color range for better detection\n";
        outFile << "- Multi-scale top-hat transforms (7x7, 15x15, 25x25)\n";
        outFile << "- Enhanced morphological operations\n";
        outFile << "- CPU-optimized implementation for Apple Silicon\n\n";
        outFile << "Descriptor values:\n";
        for (size_t i = 0; i < descriptor.size(); ++i) {
            outFile << "Angle " << i << ": " << descriptor[i] << "\n";
        }
        outFile.close();
        cout << "\nResults saved to simple_crofton_result.txt" << endl;
    }
    
    cout << "\nEnhanced edge detection completed successfully!" << endl;
    cout << "Press any key to exit..." << endl;
    
    waitKey(0);
    return 0;
}