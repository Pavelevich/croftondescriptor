#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>

// Device code shared with the webapp backend (NVRTC compiles the same file).
#include "crofton_kernels.cu"

using namespace cv;
using namespace std;

static const int CROFTON_MAX_POINTS = 239;
static const int CANT_PHI = 361;
static const float PI_F = 3.14159265358979f;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            cerr << "CUDA error " << cudaGetErrorString(err__)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ----------------------------------------------------
// Resample contour to nPoints by uniform arc length
// ----------------------------------------------------
vector<Point2f> resampleContour(const vector<Point>& contour, int nPoints) {
    vector<Point2f> resampled;
    if (contour.empty() || nPoints <= 0) return resampled;

    vector<double> cumDist;
    cumDist.push_back(0.0);
    double totalLen = 0.0;
    for (size_t i = 0; i < contour.size(); ++i) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()];
        totalLen += hypot(double(p2.x - p1.x), double(p2.y - p1.y));
        cumDist.push_back(totalLen);
    }

    if (totalLen < 1e-9) {
        resampled.resize(nPoints, Point2f(contour[0]));
        return resampled;
    }

    double step = totalLen / nPoints;
    resampled.reserve(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        double currentDist = i * step;
        auto it = std::lower_bound(cumDist.begin(), cumDist.end(), currentDist);
        int idx = int(it - cumDist.begin());
        if (idx == 0) {
            resampled.push_back(Point2f(contour[0]));
        } else {
            int prevIdx = idx - 1;
            double segLen = cumDist[idx] - cumDist[prevIdx];
            double frac = segLen > 1e-12 ? (currentDist - cumDist[prevIdx]) / segLen : 0.0;
            Point2f p1 = Point2f(contour[prevIdx]);
            Point2f p2 = Point2f(contour[idx % contour.size()]);
            resampled.push_back(p1 + float(frac) * (p2 - p1));
        }
    }
    return resampled;
}

// Exact max pairwise distance (O(N^2), N=239 — negligible)
float Sdiametro(const float* borde, int N, int realN) {
    float d = 0.0f;
    for (int i = 0; i < realN - 1; i++) {
        for (int j = i + 1; j < realN; j++) {
            float dx = borde[i] - borde[j];
            float dy = borde[N + i] - borde[N + j];
            float dist = sqrtf(dx * dx + dy * dy);
            if (d < dist) d = dist;
        }
    }
    return d;
}

// ----------------------------------------------------
// Crofton descriptor on the GPU
// ----------------------------------------------------
void croftonDescriptorGPU(const vector<float>& contourData) {
    ofstream logFile("crofton_log.txt", ios::app);

    const int N = CROFTON_MAX_POINTS;
    vector<float> hostBorde(2 * N, 0.0f);

    int realN = int(contourData.size() / 2);
    if (realN > N) realN = N;
    for (int i = 0; i < realN; i++) {
        hostBorde[i]     = contourData[i];
        hostBorde[N + i] = contourData[realN + i];
    }

    float diam = Sdiametro(hostBorde.data(), N, realN);

    // Offset window: |projection| <= max ||p|| for every angle, so 2*R always
    // covers all crossings (the pairwise diameter would clip oblique-angle
    // crossings on non-centrally-symmetric shapes).
    float radius = 0.0f;
    for (int i = 0; i < realN; ++i) {
        float r = sqrtf(hostBorde[i] * hostBorde[i]
                        + hostBorde[N + i] * hostBorde[N + i]);
        if (r > radius) radius = r;
    }
    float window = 2.0f * radius;
    int cant_p = max(int(ceilf(radius)), 1);  // binW stays ~2 px

    float *dBorde = nullptr, *dSProyX = nullptr, *dSMap = nullptr;
    float *dCurve = nullptr, *dFeret = nullptr;
    CUDA_CHECK(cudaMalloc(&dBorde, 2 * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSProyX, CANT_PHI * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSMap, cant_p * CANT_PHI * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCurve, CANT_PHI * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFeret, CANT_PHI * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dBorde, hostBorde.data(), 2 * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block2(16, 16);
    dim3 grid2((CANT_PHI + 15) / 16, (N + 15) / 16);
    proyectionKernel<<<grid2, block2>>>(dBorde, dSProyX, N, realN, CANT_PHI);

    dim3 block1(64);
    dim3 grid1((CANT_PHI + 63) / 64);
    kernelCrofton<<<grid1, block1>>>(dSProyX, dSMap, N, realN, CANT_PHI, cant_p, window);
    reduceKernel<<<grid1, block1>>>(dSProyX, dSMap, dCurve, dFeret,
                                    N, realN, CANT_PHI, cant_p, window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> hostSMap(cant_p * CANT_PHI);
    vector<float> hostCurve(CANT_PHI), hostFeret(CANT_PHI);
    CUDA_CHECK(cudaMemcpy(hostSMap.data(), dSMap, hostSMap.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostCurve.data(), dCurve, CANT_PHI * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostFeret.data(), dFeret, CANT_PHI * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(dBorde); cudaFree(dSProyX); cudaFree(dSMap);
    cudaFree(dCurve); cudaFree(dFeret);

    // Cauchy–Crofton perimeter estimate from the vote map (phi over [0, 180))
    double perimeter = 0.0;
    for (int phi = 0; phi < 180; ++phi) perimeter += hostCurve[phi];
    perimeter *= PI_F / 180.0;

    float smapMax = *max_element(hostSMap.begin(), hostSMap.end());

    logFile << "Diametro calculado: " << diam << endl;
    logFile << "SMap dimension: " << cant_p << " x " << CANT_PHI << endl;
    logFile << "SMap max crossings: " << smapMax << endl;
    logFile << "Perimetro (Cauchy-Crofton): " << perimeter << endl;
    logFile << "Descriptor C(phi):" << endl;
    for (int phi = 0; phi < CANT_PHI; ++phi)
        logFile << "Angle " << phi << ": " << hostCurve[phi]
                << "  (feret " << hostFeret[phi] << ")" << endl;
    logFile << "-------------------------------------" << endl;

    ofstream csv("crofton_descriptor.csv");
    csv << "phi_deg,crofton,feret\n";
    for (int phi = 0; phi < CANT_PHI; ++phi)
        csv << phi << "," << hostCurve[phi] << "," << hostFeret[phi] << "\n";

    cout << "Perimetro (Cauchy-Crofton GPU): " << perimeter << " px" << endl;
    cout << "SMap max crossings: " << smapMax << endl;
}

// ----------------------------------------------------
// Preprocessing: HSV mask + grayscale top-hat + morphology
// ----------------------------------------------------
Mat preprocess(const Mat& imgColor) {
    Mat hsv;
    cvtColor(imgColor, hsv, COLOR_BGR2HSV);
    Mat maskHSV;
    inRange(hsv, Scalar(100, 20, 20), Scalar(180, 255, 255), maskHSV);

    Mat imgGray;
    cvtColor(imgColor, imgGray, COLOR_BGR2GRAY);
    Mat kernelTopHat = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    Mat openedGray;
    morphologyEx(imgGray, openedGray, MORPH_OPEN, kernelTopHat);
    Mat topHat = imgGray - openedGray;

    Mat binTopHat;
    threshold(topHat, binTopHat, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat combined;
    bitwise_or(maskHSV, binTopHat, combined);

    Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat opened, closed;
    morphologyEx(combined, opened, MORPH_OPEN, kernelOpen);
    morphologyEx(opened, closed, MORPH_CLOSE, kernelClose);
    return closed;
}

int main(int argc, char** argv) {
    string imagePath = argc > 1 ? argv[1] : "test_cell.jpg";
    bool show = false;
    for (int i = 2; i < argc; ++i)
        if (string(argv[i]) == "--show") show = true;

    Mat imgColor = imread(imagePath);
    if (imgColor.empty()) {
        cerr << "Error: No se pudo cargar la imagen: " << imagePath << endl;
        return -1;
    }

    Mat closed = preprocess(imgColor);

    vector<vector<Point>> contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if (contours.empty()) {
        cerr << "No se encontraron contornos." << endl;
        return -1;
    }

    int largestIdx = 0;
    double largestArea = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > largestArea) { largestArea = area; largestIdx = int(i); }
    }
    cout << "Contornos: " << contours.size()
         << " | area mayor: " << largestArea
         << " | perimetro (OpenCV): " << arcLength(contours[largestIdx], true)
         << endl;

    Mat contourImg = imgColor.clone();
    drawContours(contourImg, contours, largestIdx, Scalar(0, 255, 0), 2);
    imwrite("contour_result.jpg", contourImg);

    vector<Point2f> resampled = resampleContour(contours[largestIdx], CROFTON_MAX_POINTS);

    // Center on the exact float bounding-box center (cv::boundingRect returns
    // an integer Rect and would quantize the center vs the webapp pipeline).
    float xmin = resampled[0].x, xmax = resampled[0].x;
    float ymin = resampled[0].y, ymax = resampled[0].y;
    for (const auto& pt : resampled) {
        xmin = min(xmin, pt.x); xmax = max(xmax, pt.x);
        ymin = min(ymin, pt.y); ymax = max(ymax, pt.y);
    }
    float cx = (xmin + xmax) * 0.5f;
    float cy = (ymin + ymax) * 0.5f;
    for (auto& pt : resampled) { pt.x -= cx; pt.y -= cy; }

    vector<float> contourData;
    contourData.reserve(resampled.size() * 2);
    for (const auto& p : resampled) contourData.push_back(p.x);
    for (const auto& p : resampled) contourData.push_back(p.y);

    cout << "\n--- Ejecutando Descriptor Crofton (CUDA) ---" << endl;
    croftonDescriptorGPU(contourData);
    cout << "Listo: contour_result.jpg, crofton_descriptor.csv, crofton_log.txt" << endl;

    if (show) {
        imshow("Contorno Detectado", contourImg);
        waitKey(0);
    }
    return 0;
}
