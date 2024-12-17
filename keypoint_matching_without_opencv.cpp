// main.cpp
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <bits/stdc++.h>
#include "stb_image.h"
#include "stb_image_write.h"

// FLANN headers
#include <flann/flann.hpp>

//--------------------------------------
// A small struct to hold image data
//--------------------------------------
struct GrayImage {
    int width;
    int height;
    std::vector<float> data; // Use float for convenient convolution ops

    GrayImage(int w=0, int h=0) : width(w), height(h), data(w*h, 0.0f) {}
};

//--------------------------------------
// Utility to load an image as grayscale
//--------------------------------------
GrayImage loadGrayImage(const std::string& filename) {
    int w,h,c;
    unsigned char* imgData = stbi_load(filename.c_str(), &w, &h, &c, 0);
    if(!imgData) {
        std::cerr << "Error: Could not load image: " << filename << std::endl;
        exit(1);
    }
    
    // Convert to single-channel grayscale if not already
    GrayImage gray(w, h);
    if(c == 1) {
        // Already grayscale
        for(int i = 0; i < w*h; i++)
            gray.data[i] = imgData[i];
    } else {
        // average R,G,B if color
        for(int i = 0; i < w*h; i++){
            float r = imgData[3*i + 0];
            float g = imgData[3*i + 1];
            float b = imgData[3*i + 2];
            gray.data[i] = 0.299f*r + 0.587f*g + 0.114f*b; 
        }
    }
    stbi_image_free(imgData);
    return gray;
}

//--------------------------------------
// Utility to save an RGB image to disk
//--------------------------------------
void saveRGBImage(const std::string& filename, const std::vector<unsigned char>& rgb,
                  int width, int height)
{
    bool success = false;
    if(filename.size() >= 4 && filename.substr(filename.size()-4)==".png") {
        success = stbi_write_png(filename.c_str(), width, height, 3, rgb.data(), width*3);
    } 
    else if(filename.size() >= 4 && filename.substr(filename.size()-4)==".jpg") {
        success = stbi_write_jpg(filename.c_str(), width, height, 3, rgb.data(), 100);
    }
    else {
        std::cerr << "Warning: unrecognized extension, saving as PNG." << std::endl;
        success = stbi_write_png(filename.c_str(), width, height, 3, rgb.data(), width*3);
    }
    if(!success) {
        std::cerr << "Error: Failed to save image: " << filename << std::endl;
    } else {
        std::cout << "Image saved: " << filename << std::endl;
    }
}

//--------------------------------------
// 2D Convolution (for float grayscale images)
//--------------------------------------
GrayImage convolve2D(const GrayImage& in, const std::vector<float>& kernel, int kW, int kH) {
    GrayImage out(in.width, in.height);
    int halfW = kW / 2;
    int halfH = kH / 2;
    for(int r = 0; r < in.height; r++){
        for(int c = 0; c < in.width; c++){
            float sum = 0.0f;
            for(int kr = -halfH; kr <= halfH; kr++){
                for(int kc = -halfW; kc <= halfW; kc++){
                    int rr = r + kr;
                    int cc = c + kc;
                    if(rr < 0 || rr >= in.height || cc < 0 || cc >= in.width) 
                        continue; 
                    float val = in.data[rr*in.width + cc];
                    float w = kernel[(kr+halfH)*kW + (kc+halfW)];
                    sum += val * w;
                }
            }
            out.data[r*out.width + c] = sum;
        }
    }
    return out;
}

//--------------------------------------
// FAST corner detection (simplified version)
//--------------------------------------
std::vector<std::pair<int,int>> FAST(const GrayImage& in, int N=9, float threshold=0.15f, int nms_window=2) {
    // small Gaussian blur: [[1,2,1],[2,4,2],[1,2,1]] / 16
    std::vector<float> kernel = {1,2,1,  2,4,2,  1,2,1};
    for(auto &k : kernel) k /= 16.0f;
    GrayImage img = convolve2D(in, kernel, 3, 3);

    // Cross idx & circle idx
    static int cross_idx[8] = {3,0, -3,0,  0,3, 0,-3}; 
    static int circle_idx[32] = {
        3,3, 2,1, 0,-1, -2,-3, -3,-3, -2,-1, 0,1, 2,3,
        0,1, 2,3, 3,3,  2,1,  0,-1, -2,-3, -3,-3, -2,-1
    };
    GrayImage cornerScore(img.width, img.height);

    std::vector<std::pair<int,int>> keypoints;
    keypoints.reserve(img.width*img.height / 100); // just a guess

    for(int y=3; y < img.height-3; y++){
        for(int x=3; x < img.width-3; x++){
            float Ip = img.data[y*img.width + x];
            float t = (threshold < 1.0f) ? threshold * Ip : threshold; 
            // Cross check
            int countBright = 0;
            int countDark   = 0;
            for(int k=0; k<4; k++){
                int yy = y + cross_idx[2*k+1];
                int xx = x + cross_idx[2*k];
                float val = img.data[yy*img.width + xx];
                if(val > Ip + t) countBright++;
                if(val < Ip - t) countDark++;
            }
            if(countBright >= 3 || countDark >= 3){
                // Full circle check
                int brightCircle = 0;
                int darkCircle   = 0;
                for(int cidx=0; cidx<16; cidx++){
                    int yy = y + circle_idx[cidx+16];
                    int xx = x + circle_idx[cidx];
                    float val = img.data[yy*img.width + xx];
                    if(val >= Ip + t) brightCircle++;
                    if(val <= Ip - t) darkCircle++;
                }
                if(brightCircle >= N || darkCircle >= N) {
                    keypoints.push_back({x,y});
                    // corner score
                    float score = 0.0f;
                    for(int cidx=0; cidx<16; cidx++){
                        int yy = y + circle_idx[cidx+16];
                        int xx = x + circle_idx[cidx];
                        float val = img.data[yy*img.width + xx];
                        score += std::fabs(Ip - val);
                    }
                    cornerScore.data[y*cornerScore.width + x] = score;
                }
            }
        }
    }

    // NMS
    std::vector<std::pair<int,int>> finalKps;
    finalKps.reserve(keypoints.size());
    if(nms_window != 0){
        std::unordered_set<long long> used;
        auto idx2D = [&](int xx, int yy){ return (long long)yy*cornerScore.width + xx; };

        for(auto& kp : keypoints){
            int x = kp.first;
            int y = kp.second;
            if(used.find(idx2D(x,y)) != used.end()) continue;
            float bestScore = cornerScore.data[y*cornerScore.width + x];
            int bestX = x, bestY = y;
            for(int ry=-nms_window; ry<=nms_window; ry++){
                for(int rx=-nms_window; rx<=nms_window; rx++){
                    int ny = y + ry, nx = x + rx;
                    if(ny<0 || ny>=cornerScore.height || nx<0 || nx>=cornerScore.width) continue;
                    float sc = cornerScore.data[ny*cornerScore.width + nx];
                    if(sc > bestScore){
                        bestScore = sc;
                        bestX = nx;
                        bestY = ny;
                    }
                }
            }
            if(used.find(idx2D(bestX,bestY)) == used.end()){
                finalKps.push_back({bestX,bestY});
                used.insert(idx2D(bestX,bestY));
            }
        }
    } else {
        finalKps = keypoints;
    }

    return finalKps;
}

//--------------------------------------
// Orientation (similar to Python corner_orientations)
//--------------------------------------
std::vector<float> cornerOrientations(const GrayImage& in, const std::vector<std::pair<int,int>>& corners) {
    // Build the 31x31 mask
    std::vector<int> OFAST_UMAX = {15,15,15,15,14,14,14,13,13,12,11,10,9,8,6,3};
    const int maskSize = 31;
    std::vector<int> mask(maskSize*maskSize, 0);
    for(int i=-15; i<=15; i++){
        int row = i + 15;
        int colRange = OFAST_UMAX[std::abs(i)];
        for(int j=-colRange; j<=colRange; j++){
            int col = j + 15;
            mask[row*maskSize + col] = 1;
        }
    }

    // pad the input image
    int pad = 15;
    GrayImage padded(in.width + 2*pad, in.height + 2*pad);
    for(int r=0; r<in.height; r++){
        for(int c=0; c<in.width; c++){
            padded.data[(r+pad)*padded.width + (c+pad)] = in.data[r*in.width + c];
        }
    }

    std::vector<float> orientations;
    orientations.reserve(corners.size());

    for(auto &pt : corners){
        int c0 = pt.first, r0 = pt.second;
        int rOff = r0 + pad;
        int cOff = c0 + pad;
        float m01 = 0.f, m10 = 0.f;

        for(int rr=0; rr<maskSize; rr++){
            float rowSum = 0.0f;
            for(int cc=0; cc<maskSize; cc++){
                if(mask[rr*maskSize + cc] == 0) continue;
                float I = padded.data[(rOff + rr - 15)*padded.width + (cOff + cc - 15)];
                // m10 accumulation
                m10 += I * (cc - 15);
                rowSum += I;
            }
            m01 += rowSum * (rr - 15);
        }
        float angle = std::atan2(m01, m10);
        orientations.push_back(angle);
    }
    return orientations;
}

//--------------------------------------
// BRIEF descriptor
//--------------------------------------
std::vector<std::vector<bool>> BRIEF(const GrayImage& in, 
                                     const std::vector<std::pair<int,int>>& keypoints,
                                     const std::vector<float>* orientations = nullptr,
                                     int n=256, int patch_size=9, 
                                     float sigma=1.0f, 
                                     const std::string& mode="uniform", 
                                     int sample_seed=42)
{
    // 5x5 Gaussian kernel
    std::vector<float> kernel5 = {
        1,   4,   7,   4,  1,
        4,  16,  26,  16,  4,
        7,  26,  41,  26,  7,
        4,  16,  26,  16,  4,
        1,   4,   7,   4,  1
    };
    for(auto &k : kernel5) k /= 273.f;
    GrayImage img = convolve2D(in, kernel5, 5, 5);

    // random sampling pairs
    std::mt19937 rng(sample_seed);
    std::uniform_int_distribution<int> uniformDist(-(patch_size-2)/2 +1, (patch_size/2));

    std::vector<std::pair<int,int>> pos1(n), pos2(n);
    for(int i=0; i<n; i++){
        pos1[i] = {uniformDist(rng), uniformDist(rng)};
        pos2[i] = {uniformDist(rng), uniformDist(rng)};
    }

    std::vector<std::vector<bool>> descriptors; 
    descriptors.reserve(keypoints.size());

    bool useOrientation = (orientations != nullptr) && (orientations->size() == keypoints.size());
    int border = (useOrientation) ? (int)((patch_size/2)*1.5f) : (patch_size/2);

    for(int i=0; i<(int)keypoints.size(); i++){
        int c0 = keypoints[i].first;
        int r0 = keypoints[i].second;
        if(r0 < border || r0 >= img.height - border || c0 < border || c0 >= img.width - border){
            continue;
        }

        std::vector<bool> desc(n, false);
        float angle = 0.f, sin_th=0.f, cos_th=1.f;
        if(useOrientation){
            angle   = (*orientations)[i];
            sin_th  = std::sin(angle);
            cos_th  = std::cos(angle);
        }

        for(int p=0; p<n; p++){
            int pr0 = pos1[p].first, pc0 = pos1[p].second;
            int pr1 = pos2[p].first, pc1 = pos2[p].second;

            int rr0, cc0, rr1, cc1;
            if(useOrientation){
                rr0 = r0 + (int)std::round(sin_th*pr0 + cos_th*pc0);
                cc0 = c0 + (int)std::round(cos_th*pr0 - sin_th*pc0);
                rr1 = r0 + (int)std::round(sin_th*pr1 + cos_th*pc1);
                cc1 = c0 + (int)std::round(cos_th*pr1 - sin_th*pc1);
            } else {
                rr0 = r0 + pr0;  cc0 = c0 + pc0;
                rr1 = r0 + pr1;  cc1 = c0 + pc1;
            }
            float val0 = img.data[rr0*img.width + cc0];
            float val1 = img.data[rr1*img.width + cc1];
            desc[p] = (val0 < val1);
        }
        descriptors.push_back(desc);
    }

    return descriptors;
}

//--------------------------------------
// Convert binary descriptors to FLANN-friendly float matrix
// (each bit is stored as 0.0f or 1.0f, L1 distance ~ Hamming)
//--------------------------------------
flann::Matrix<float> convertDescriptorsToFlann(const std::vector<std::vector<bool>>& descriptors)
{
    size_t n = descriptors.size();
    if(n == 0) {
        return flann::Matrix<float>(nullptr, 0, 0);
    }
    size_t dim = descriptors[0].size();
    
    float* data = new float[n * dim];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            data[i*dim + j] = descriptors[i][j] ? 1.0f : 0.0f;
        }
    }
    flann::Matrix<float> mat(data, n, dim);
    return mat;
}

//--------------------------------------
// Matching with FLANN approximate KNN
// cross_check + ratio_test
//--------------------------------------
std::vector<std::pair<int,int>> matchFLANN(const std::vector<std::vector<bool>>& desc1,
                                           const std::vector<std::vector<bool>>& desc2,
                                           bool cross_check = true,
                                           float distance_ratio = 0.f)
{
    flann::Matrix<float> flannDesc1 = convertDescriptorsToFlann(desc1);
    flann::Matrix<float> flannDesc2 = convertDescriptorsToFlann(desc2);
    if (flannDesc1.rows == 0 || flannDesc2.rows == 0) {
        return {};
    }

    // Build FLANN index on desc2
    typedef flann::L1<float> Distance; // L1 for {0,1} => Hamming
    flann::Index<Distance> index2(flannDesc2, flann::KDTreeIndexParams(4));
    index2.buildIndex();

    // For ratio test, we want k=2 neighbors; otherwise k=1 is enough
    int K = (distance_ratio > 0.f) ? 2 : 1;

    std::vector<std::pair<int,int>> forwardMatches(flannDesc1.rows, {-1,-1});
    std::vector<std::vector<int>> indices(flannDesc1.rows, std::vector<int>(K));
    std::vector<std::vector<float>> dists(flannDesc1.rows, std::vector<float>(K));

    flann::SearchParams sparams;
    sparams.checks = 128; // Adjust for more accuracy vs speed

    // KNN search from desc1 to desc2
    index2.knnSearch(flannDesc1, indices, dists, K, sparams);

    // ratio test
    for (size_t i = 0; i < flannDesc1.rows; i++) {
        // if ratio test requested
        if(K == 2 && dists[i][1] > 1e-6f) {
            float ratio = dists[i][0] / dists[i][1];
            if(ratio > distance_ratio) {
                // reject
                forwardMatches[i] = {-1, -1};
                continue;
            }
        }
        forwardMatches[i] = {(int)i, indices[i][0]};
    }

    if(!cross_check){
        // gather valid matches
        std::vector<std::pair<int,int>> matches;
        matches.reserve(flannDesc1.rows);
        for(size_t i=0; i<flannDesc1.rows; i++){
            if(forwardMatches[i].second >= 0) {
                matches.push_back(forwardMatches[i]);
            }
        }
        // sort by distance
        std::vector<float> distVals(matches.size());
        for(size_t m=0; m<matches.size(); m++){
            distVals[m] = dists[matches[m].first][0];
        }
        std::vector<int> idxOrder(matches.size());
        std::iota(idxOrder.begin(), idxOrder.end(), 0);
        std::sort(idxOrder.begin(), idxOrder.end(), [&](int a, int b){
            return distVals[a] < distVals[b];
        });
        std::vector<std::pair<int,int>> sortedMatches;
        sortedMatches.reserve(matches.size());
        for(auto idx: idxOrder){
            sortedMatches.push_back(matches[idx]);
        }
        // cleanup
        delete[] flannDesc1.ptr();
        delete[] flannDesc2.ptr();
        return sortedMatches;
    }
    // cross-check
    // Build index for desc1
    flann::Index<Distance> index1(flannDesc1, flann::KDTreeIndexParams(4));
    index1.buildIndex();

    // single neighbor search from desc2->desc1
    std::vector<std::vector<int>> indices2(flannDesc2.rows, std::vector<int>(1));
    std::vector<std::vector<float>> dists2(flannDesc2.rows, std::vector<float>(1));
    index1.knnSearch(flannDesc2, indices2, dists2, 1, sparams);

    // gather consistent matches
    std::vector<std::pair<int,int>> matches;
    matches.reserve(flannDesc1.rows);

    for(size_t i=0; i<flannDesc1.rows; i++){
        if(forwardMatches[i].second < 0) continue;
        int idx2 = forwardMatches[i].second;
        // cross-check condition
        if(indices2[idx2][0] == (int)i) {
            matches.push_back(forwardMatches[i]);
        }
    }

    // sort by distance
    std::vector<float> distVals(matches.size());
    for(size_t m=0; m<matches.size(); m++){
        distVals[m] = dists[matches[m].first][0];
    }
    std::vector<int> idxOrder(matches.size());
    std::iota(idxOrder.begin(), idxOrder.end(), 0);
    std::sort(idxOrder.begin(), idxOrder.end(), [&](int a, int b){
        return distVals[a] < distVals[b];
    });

    std::vector<std::pair<int,int>> sortedMatches;
    sortedMatches.reserve(matches.size());
    for(auto idx: idxOrder){
        sortedMatches.push_back(matches[idx]);
    }

    // cleanup
    delete[] flannDesc1.ptr();
    delete[] flannDesc2.ptr();

    return sortedMatches;
}

//--------------------------------------
// Draw top matches in a side-by-side RGB image
//--------------------------------------
std::vector<unsigned char> drawMatches(
    const GrayImage& left, 
    const GrayImage& right,
    const std::vector<std::pair<int,int>>& topMatches,
    const std::vector<std::pair<int,int>>& cornersLeft,
    const std::vector<std::pair<int,int>>& cornersRight,
    int numMatchesToDraw=3)
{
    int outW = left.width + right.width;
    int outH = std::max(left.height, right.height);
    std::vector<unsigned char> outRGB(outW*outH*3, 0);

    // Copy left grayscale to RGB
    for(int r=0; r<left.height; r++){
        for(int c=0; c<left.width; c++){
            int idxOut = (r*outW + c)*3;
            unsigned char g = (unsigned char)std::max(0.f, std::min(255.f, left.data[r*left.width + c]));
            outRGB[idxOut + 0] = g;
            outRGB[idxOut + 1] = g;
            outRGB[idxOut + 2] = g;
        }
    }
    // Copy right grayscale to RGB
    for(int r=0; r<right.height; r++){
        for(int c=0; c<right.width; c++){
            int idxOut = (r*outW + (c+left.width))*3;
            unsigned char g = (unsigned char)std::max(0.f, std::min(255.f, right.data[r*right.width + c]));
            outRGB[idxOut + 0] = g;
            outRGB[idxOut + 1] = g;
            outRGB[idxOut + 2] = g;
        }
    }

    // Draw lines for top 'numMatchesToDraw'
    int drawCount = std::min<int>(numMatchesToDraw, (int)topMatches.size());
    for(int i=0; i<drawCount; i++){
        auto match = topMatches[i];
        int idxLeft  = match.first;
        int idxRight = match.second;
        if(idxLeft < 0 || idxLeft >= (int)cornersLeft.size()) continue;
        if(idxRight< 0 || idxRight>= (int)cornersRight.size()) continue;

        int xL = cornersLeft[idxLeft].first;
        int yL = cornersLeft[idxLeft].second;
        int xR = cornersRight[idxRight].first + left.width;
        int yR = cornersRight[idxRight].second;

        auto drawPixel = [&](int rr, int cc){
            if(rr<0 || rr>=outH || cc<0 || cc>=outW) return;
            int idxRGB = (rr*outW + cc)*3;
            outRGB[idxRGB + 0] = 0;
            outRGB[idxRGB + 1] = 255;
            outRGB[idxRGB + 2] = 0;
        };

        // Bresenham
        int dx = std::abs(xR - xL), sx = (xL < xR)? 1 : -1;
        int dy = -std::abs(yR - yL), sy = (yL < yR)? 1 : -1;
        int err = dx + dy;
        int cx = xL, cy = yL;
        while(true){
            for(int ry=-1; ry<=1; ry++){
                for(int rx=-1; rx<=1; rx++){
                    drawPixel(cy+ry, cx+rx);
                }
            }
            if(cx == xR && cy == yR) break;
            int e2 = 2*err;
            if(e2 >= dy){ err += dy; cx += sx; }
            if(e2 <= dx){ err += dx; cy += sy; }
        }
    }
    return outRGB;
}

//--------------------------------------
// Main
//--------------------------------------
int main()
{
    // Paths to your images
    std::string leftImagePath  = "../Datasets/courtyard_dslr_undistorted/courtyard/images/left.jpg";
    std::string rightImagePath = "../Datasets/courtyard_dslr_undistorted/courtyard/images/right.jpg";

    GrayImage leftImg  = loadGrayImage(leftImagePath);
    GrayImage rightImg = loadGrayImage(rightImagePath);

    // 1. Detect FAST corners
    auto cornersLeft  = FAST(leftImg,  9, 0.15f, 2);
    auto cornersRight = FAST(rightImg, 9, 0.15f, 2);

    // 2. Compute orientation (O-FAST style)
    auto orientsLeft  = cornerOrientations(leftImg, cornersLeft);
    auto orientsRight = cornerOrientations(rightImg, cornersRight);

    // 3. Extract BRIEF descriptors
    auto descLeft  = BRIEF(leftImg, cornersLeft, &orientsLeft, 512, 8, 1.0f, "uniform");
    auto descRight = BRIEF(rightImg, cornersRight, &orientsRight, 512, 8, 1.0f, "uniform");

    // 4. FLANN-based matching with cross-check & optional ratio test
    float ratioTest = 0.7f; // e.g. Lowe’s ratio
    auto matches = matchFLANN(descLeft, descRight, true, ratioTest);
    // Now matches are sorted by ascending distance

    // 5. Draw top matches
    int numMatchesToDraw = 3;
    auto outRGB = drawMatches(leftImg, rightImg, matches, cornersLeft, cornersRight, numMatchesToDraw);

    // 6. Save result
    std::string outputPath = "output_without_opencv.jpg";
    int outW = leftImg.width + rightImg.width;
    int outH = std::max(leftImg.height, rightImg.height);
    saveRGBImage(outputPath, outRGB, outW, outH);

    return 0;
}

