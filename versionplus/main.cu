#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// --------------------------------------------------------------------------------
// Reflection on 5–7 Potential Sources of Error for Missing the Outer Transparent Edge:
//
// 1) HSV color range might be too narrow or the outer ring is nearly background hue.
// 2) Morphological operations might remove that faint boundary if it's too small.
// 3) The boundingRect + centering might cause offset issues if the contour is noisy.
// 4) The shape's lighting or background might require top-hat or alternative approach
//    specifically for faint transitions (semi-transparent edges).
// 5) BANDA in kernelCrofton could be too small if the diameter is large, leading to 0 in SMap.
// 6) We might need multi-scale or difference-of-Gaussians in grayscale to see faint edges.
// 7) The outer region might not have enough saturation or value to appear in the HSV mask.
//
// Distilled to 1–2 Likely Issues:
// A) The outer ring is almost the same color as the background, so HSV alone misses it.
// B) A top-hat transform on grayscale could highlight that faint ring more effectively.
//
// Logs to validate assumptions:
//  - Pixel counts in the top-hat result, plus morphological steps
//  - Average HSV in the largest contour (to see if it’s near background hue/sat).
//
// Then we attempt a new partial fix:
//  - Apply top-hat in grayscale to highlight faint boundaries
//  - Combine that top-hat result with the widened HSV mask (logical OR)
//  - Then do a mild morphology, findContours, and proceed as before.
// --------------------------------------------------------------------------------

using namespace cv;
using namespace std;

// Configuration for Crofton
static const int CROFTON_MAX_POINTS = 239;
static const float PI = 3.1415927f;

// ----------------------------------------------------
// 1) Resample Contour to nPoints
// ----------------------------------------------------
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
        // Degenerate
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

// ----------------------------------------------------
// 2) Crofton descriptor helpers
// ----------------------------------------------------
float Sdiametro(const float* borde, int N) {
    float d=0.0f;
    for (int i=0; i<N-1; i++){
        for (int j=i+1; j<N; j++){
            float dist_X = powf((borde[i] - borde[j]), 2);
            float dist_Y = powf((borde[N + i] - borde[N + j]), 2);
            float dist_temp = sqrtf(dist_X + dist_Y);
            if(d < dist_temp) d=dist_temp;
        }
    }
    return d;
}

void SproyectX(const float* borde, int N, int phi, float *SProyX) {
    for (int i=0; i<phi; i++){
        float angulo = (i*PI)/180.0f;
        for (int j=0; j<N; j++){
            float x = borde[j];
            float y = borde[N + j];
            SProyX[i*N + j] = x*cosf(angulo) + y*sinf(angulo);
        }
    }
}

// ----------------------------------------------------
// 3) __device__ and kernel for Crofton
// ----------------------------------------------------
__device__ float proyectY(float x,float y,float angle) {
    return x*sinf(-angle) + y*cosf(angle);
}

static const float BANDA = 10.0f;  // Keep or adjust if needed
static const float EPS   = 1e-5f;

__device__ void interpolar(...) { /* Your device code here */ }
__device__ void screateListEtq(...) { /* Your device code here */ }
__device__ void orden(...) { /* Your device code here */ }

__global__ void kernelCrofton(float *dBorde, float *dSProyX, float *dSMap) {
    // Your kernel code
}

// ----------------------------------------------------
// 4) Descriptor Crofton in GPU with logs
// ----------------------------------------------------
void croftonDescriptorGPU(const vector<float> &contourData) {
    ofstream logFile("crofton_log.txt", ios::app);
    if(!logFile.is_open()){
        cerr << "Error: no se pudo abrir crofton_log.txt" << endl;
        return;
    }

    static const int N = CROFTON_MAX_POINTS;
    float hostBorde[2*N];
    memset(hostBorde, 0, sizeof(hostBorde));

    int realNumPoints = int(contourData.size()/2);
    if(realNumPoints > N) realNumPoints = N;

    for(int i=0; i<realNumPoints; i++){
        hostBorde[i]       = contourData[i];                 // X
        hostBorde[N + i]   = contourData[realNumPoints + i]; // Y
    }

    float diam = Sdiametro(hostBorde, N);
    int cant_phi = 361;
    int cant_p   = int(ceil(diam/2.0f));

    vector<float> hostSProyX(cant_phi*N, 0.0f);
    vector<float> hostSMap(cant_p*cant_phi, 0.0f);

    SproyectX(hostBorde, N, cant_phi, hostSProyX.data());

    float *dBorde=nullptr, *dSProyX=nullptr, *dSMap=nullptr;
    cudaMalloc((void**)&dBorde,  2*N*sizeof(float));
    cudaMalloc((void**)&dSProyX, cant_phi*N*sizeof(float));
    cudaMalloc((void**)&dSMap,   cant_p*cant_phi*sizeof(float));

    cudaMemcpy(dBorde,  hostBorde,       2*N*sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(dSProyX, hostSProyX.data(), cant_phi*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dSMap,   0, cant_p*cant_phi*sizeof(float));

    dim3 blockDim(4);
    dim3 gridDim((cant_phi + blockDim.x -1)/blockDim.x);
    kernelCrofton<<<gridDim, blockDim>>>(dBorde, dSProyX, dSMap);
    cudaDeviceSynchronize();

    cudaMemcpy(hostSMap.data(), dSMap, cant_p*cant_phi*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dBorde);
    cudaFree(dSProyX);
    cudaFree(dSMap);

    logFile << "Diametro calculado: " << diam << endl;
    logFile << "SMap dimension: " << cant_p << " x " << cant_phi << endl;
    logFile << "Primeros valores de SMap: ";
    for(int i=0; i<min(cant_p*cant_phi, 10); i++){
        logFile << hostSMap[i] << " ";
    }
    logFile << "\n-------------------------------------\n";
    logFile.close();
}

// ----------------------------------------------------
// 5) MAIN with logs + partial “fix”: Top-Hat in grayscale + HSV mask
// ----------------------------------------------------
int main() {
    // Reflection on 5–7 potential issues:
    // (listed above), we suspect we need a top-hat for the faint ring.

    // 1) Load color image
    Mat imgColor = imread("/home/pavel/Downloads/Telegram Desktop/photo_2025-03-16_19-35-40.jpg");
    if(imgColor.empty()){
        cerr << "Error: No se pudo cargar la imagen.\n";
        return -1;
    }
    imshow("Imagen Original", imgColor);

    // 2) Convert to HSV for color-based mask
    Mat hsv;
    cvtColor(imgColor, hsv, COLOR_BGR2HSV);

    // Widen HSV range a bit
    Scalar lowerPurple(100, 20, 20);
    Scalar upperPurple(180, 255, 255);
    Mat maskHSV;
    inRange(hsv, lowerPurple, upperPurple, maskHSV);

    // LOG: White pixels in HSV mask BEFORE morph
    int beforeMorphHSV = countNonZero(maskHSV);
    cout << "[LOG] White pixels in HSV mask BEFORE morph: " << beforeMorphHSV << endl;

    // 3) Top-Hat transform on grayscale to highlight faint edges
    Mat imgGray;
    cvtColor(imgColor, imgGray, COLOR_BGR2GRAY);

    // We'll do a morphological top-hat: top-hat = original - open(original)
    // This can highlight bright regions smaller than structuring element
    Mat kernelTopHat = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    Mat openedGray, topHat;
    morphologyEx(imgGray, openedGray, MORPH_OPEN, kernelTopHat);
    topHat = imgGray - openedGray; // highlight bright structures

    // Binarize top-hat with Otsu or fixed threshold
    Mat binTopHat;
    threshold(topHat, binTopHat, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Log how many white pixels in topHat
    int whiteTopHat = countNonZero(binTopHat);
    cout << "[LOG] White pixels in topHat bin: " << whiteTopHat << endl;

    // 4) Combine HSV mask with top-hat result (logical OR)
    // This might capture both color-based region AND faint grayscale edges
    Mat combined;
    bitwise_or(maskHSV, binTopHat, combined);

    // 5) Morphology on combined (two-step: open then close)
    Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    Mat opened, closed;

    morphologyEx(combined, opened, MORPH_OPEN, kernelOpen, Point(-1,-1), 1);
    morphologyEx(opened, closed, MORPH_CLOSE, kernelClose, Point(-1,-1), 1);

    int afterOpen = countNonZero(opened);
    int afterClose = countNonZero(closed);
    cout << "[LOG] White after open: " << afterOpen
         << ", after close: " << afterClose << endl;

    imshow("Mask HSV", maskHSV);
    imshow("TopHat", topHat);
    imshow("TopHat Bin", binTopHat);
    imshow("Combined", combined);
    imshow("Combined Opened", opened);
    imshow("Combined Closed", closed);

    // 6) findContours on closed
    vector<vector<Point>> contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if(contours.empty()){
        cerr << "No se encontraron contornos.\n";
        waitKey(0);
        return -1;
    }

    // 7) Largest contour
    int largestIdx = 0;
    double largestArea = 0.0;
    for(size_t i=0; i<contours.size(); i++){
        double area = contourArea(contours[i]);
        if(area > largestArea){
            largestArea = area;
            largestIdx = (int)i;
        }
    }

    // 8) Log average HSV in largest contour
    double totalH=0, totalS=0, totalV=0;
    int countHSV=0;
    for(const auto &pt : contours[largestIdx]) {
        if(pt.x >= 0 && pt.x < hsv.cols && pt.y >= 0 && pt.y < hsv.rows) {
            Vec3b hsvPixel = hsv.at<Vec3b>(pt.y, pt.x);
            totalH += hsvPixel[0];
            totalS += hsvPixel[1];
            totalV += hsvPixel[2];
            countHSV++;
        }
    }
    if(countHSV>0){
        double avgH = totalH/countHSV;
        double avgS = totalS/countHSV;
        double avgV = totalV/countHSV;
        cout << "[LOG] Average H,S,V in largestContour = "
             << avgH << ", " << avgS << ", " << avgV << endl;
    } else {
        cout << "[LOG] Could not compute average HSV (count=0)\n";
    }

    // 9) Visualize largest contour
    Mat contourImg = imgColor.clone();
    drawContours(contourImg, contours, largestIdx, Scalar(0,255,0), 2);
    imshow("Contorno Detectado", contourImg);

    // 10) Resample to 239
    vector<Point> resampled = resampleContour(contours[largestIdx], CROFTON_MAX_POINTS);

    // 11) Center
    Rect box = boundingRect(resampled);
    float cx = box.x + box.width*0.5f;
    float cy = box.y + box.height*0.5f;
    for(auto &pt : resampled){
        pt.x = cvRound(pt.x - cx);
        pt.y = cvRound(pt.y - cy);
    }

    // 12) Convert to [x..., y...] and call Crofton
    vector<float> contourData;
    contourData.reserve(resampled.size()*2);
    for(const Point &p : resampled){
        contourData.push_back((float)p.x);
    }
    for(const Point &p : resampled){
        contourData.push_back((float)p.y);
    }

    cout << "\n--- Ejecutando Descriptor Crofton (CUDA) ---" << endl;
    croftonDescriptorGPU(contourData);

    cout << "Proceso finalizado. Revisa crofton_log.txt para ver el descriptor.\n";

    waitKey(0);
    return 0;
}
