#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/SVD>

/**
 * @brief Normalize points for numeric stability in fundamental matrix estimation
 */
inline void normalizePoints(
    const std::vector<cv::Point2f>& points, 
    std::vector<cv::Point2f>& normalizedPoints,
    cv::Mat& T)
{
    cv::Point2f center(0,0);
    for(const auto& p: points) {
        center += p;
    }
    center *= (1.0 / points.size());

    double avgDist = 0.0;
    for(const auto& p: points) {
        avgDist += cv::norm(p - center);
    }
    avgDist /= points.size();

    double scaleFactor = std::sqrt(2.) / avgDist;

    T = (cv::Mat_<double>(3,3) << 
         scaleFactor, 0, -scaleFactor*center.x,
         0, scaleFactor, -scaleFactor*center.y,
         0, 0, 1);

    normalizedPoints.resize(points.size());
    for(size_t i=0; i<points.size(); i++){
        cv::Mat tmp = (cv::Mat_<double>(3,1) << points[i].x, points[i].y, 1.0);
        cv::Mat out = T*tmp;
        normalizedPoints[i].x = out.at<double>(0)/out.at<double>(2);
        normalizedPoints[i].y = out.at<double>(1)/out.at<double>(2);
    }
}

/**
 * @brief Estimate fundamental matrix from point correspondences (linear method)
 */
inline cv::Mat computeFundamentalMatrix(
    const std::vector<cv::Point2f> &pointsLeft,
    const std::vector<cv::Point2f> &pointsRight)
{
    if(pointsLeft.size() < 8) {
        std::cerr << "[computeFundamentalMatrix] Not enough points!\n";
        return cv::Mat();
    }

    // 1) Normalize
    std::vector<cv::Point2f> normLeft, normRight;
    cv::Mat T1, T2;
    normalizePoints(pointsLeft,  normLeft,  T1);
    normalizePoints(pointsRight, normRight, T2);

    // 2) Build matrix A
    const size_t N = normLeft.size();
    Eigen::MatrixXd A(N, 9);
    for(size_t i=0; i<N; i++){
        double x1 = normLeft[i].x,  y1 = normLeft[i].y;
        double x2 = normRight[i].x, y2 = normRight[i].y;
        A(i,0) = x2*x1;
        A(i,1) = x2*y1;
        A(i,2) = x2;
        A(i,3) = y2*x1;
        A(i,4) = y2*y1;
        A(i,5) = y2;
        A(i,6) = x1;
        A(i,7) = y1;
        A(i,8) = 1.0;
    }

    // 3) SVD on A
    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if(svdA.matrixV().cols() < 9) {
        std::cerr << "[computeFundamentalMatrix] SVD degenerate!\n";
        return cv::Mat();
    }
    Eigen::VectorXd f = svdA.matrixV().col(8); // last col

    // 4) Construct F
    cv::Mat F(3,3, CV_64F);
    for(int i=0; i<9; i++){
        F.at<double>(i/3, i%3) = f(i);
    }

    // 5) Enforce rank 2
    cv::SVD svdF(F);
    cv::Mat W = svdF.w;
    W.at<double>(2,0) = 0.0; // force rank=2
    F = svdF.u * cv::Mat::diag(W) * svdF.vt;

    // 6) Denormalize
    F = T2.t() * F * T1;
    return F;
}

/**
 * @brief RANSAC for fundamental matrix
 */
inline cv::Mat ransacFundamentalMatrix(
    const std::vector<cv::Point2f> &pointsLeft,
    const std::vector<cv::Point2f> &pointsRight,
    int iterations = 1000,
    double threshold = 0.5)
{
    if(pointsLeft.size() < 8 || pointsLeft.size() != pointsRight.size()){
        std::cerr << "[ransacFundamentalMatrix] Not enough or mismatched points!\n";
        return cv::Mat();
    }

    int maxInliers = -1;
    cv::Mat bestF;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(pointsLeft.size()-1));

    for(int iter=0; iter<iterations; iter++){
        // 1) pick 8 random distinct indices
        std::set<int> used;
        while((int)used.size()<8){
            used.insert(dis(gen));
        }
        // build subset
        std::vector<cv::Point2f> subsetL, subsetR;
        for(auto &idx: used){
            subsetL.push_back(pointsLeft[idx]);
            subsetR.push_back(pointsRight[idx]);
        }
        // 2) estimate F
        cv::Mat Fcand = computeFundamentalMatrix(subsetL, subsetR);
        if(Fcand.empty()) 
            continue;
        // 3) count inliers
        int inlierCount=0;
        for(size_t i=0; i<pointsLeft.size(); i++){
            cv::Mat x1 = (cv::Mat_<double>(3,1) << pointsLeft[i].x, pointsLeft[i].y, 1.0);
            cv::Mat x2 = (cv::Mat_<double>(3,1) << pointsRight[i].x, pointsRight[i].y, 1.0);
            double err = std::abs(x2.dot(Fcand*x1));
            if(err < threshold) inlierCount++;
        }
        // 4) update best
        if(inlierCount>maxInliers){
            maxInliers = inlierCount;
            bestF = Fcand.clone();
        }
    }

    std::cout << "[RANSAC] Max inliers: " << maxInliers 
              << " / " << pointsLeft.size() << std::endl;
    return bestF;
}

/**
 * @brief Compute sum-of-squared epipolar error for a given F
 */
inline double computeReprojectionError(
    const std::vector<cv::Point2f> &pointsLeft,
    const std::vector<cv::Point2f> &pointsRight,
    const cv::Mat &F)
{
    if(pointsLeft.size()!= pointsRight.size()) {
        std::cerr << "[computeReprojectionError] Mismatched points!\n";
        return -1;
    }
    double sumSq=0.0;
    for(size_t i=0; i<pointsLeft.size(); i++){
        cv::Mat x1 = (cv::Mat_<double>(3,1) << pointsLeft[i].x, pointsLeft[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3,1) << pointsRight[i].x, pointsRight[i].y, 1.0);
        double e = std::abs(x2.dot(F*x1));
        sumSq += e*e;
    }
    return sumSq;
}

/**
 * @brief Rectify images given camera intrinsics + baseline
 */
inline cv::Mat rectifyStereoImages(
    const cv::Mat& imgLeft, 
    const cv::Mat& imgRight,
    cv::Mat& rectifiedLeft, 
    cv::Mat& rectifiedRight,
    const cv::Mat& cameraMatrix0, 
    const cv::Mat& cameraMatrix1,
    double baseline, 
    int width,
    int height)
{
    // We create a translation vector T with baseline scaled in Y
    cv::Mat T = cv::Mat::zeros(3,1, CV_64F);
    T.at<double>(1,0) = baseline * cameraMatrix0.at<double>(0,0);

    // stereoRectify
    cv::Mat R1 = cv::Mat::zeros(3,3, CV_64F);
    cv::Mat R2 = cv::Mat::zeros(3,3, CV_64F);
    cv::Mat P1 = cv::Mat::zeros(3,4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3,4, CV_64F);
    cv::Mat Q  = cv::Mat::zeros(4,4, CV_64F);
    cv::Rect roi[2];

    cv::stereoRectify(
        cameraMatrix0, cv::Mat::zeros(1,4, CV_64F),
        cameraMatrix1, cv::Mat::zeros(1,4, CV_64F),
        cv::Size(width, height),
        cv::Mat::eye(3,3, CV_64F),
        T,
        R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, 
        -1,
        cv::Size(),
        &roi[0], &roi[1]
    );

    // initUndistortRectifyMap
    cv::Mat mapLeftX, mapLeftY, mapRightX, mapRightY;
    cv::initUndistortRectifyMap(
        cameraMatrix0, cv::Mat::zeros(1,4, CV_64F),
        R1, P1, 
        cv::Size(width, height), 
        CV_16SC2,
        mapLeftX, mapLeftY
    );
    cv::initUndistortRectifyMap(
        cameraMatrix1, cv::Mat::zeros(1,4, CV_64F),
        R2, P2,
        cv::Size(width, height),
        CV_16SC2,
        mapRightX, mapRightY
    );

    // Remap
    cv::remap(imgLeft,  rectifiedLeft,  mapLeftX,  mapLeftY,  cv::INTER_LINEAR);
    cv::remap(imgRight, rectifiedRight, mapRightX, mapRightY, cv::INTER_LINEAR);

    return Q; // Might be used for depth
}

/**
 * @brief Display or save a concatenation of rectifiedLeft and rectifiedRight
 */
inline void showRectifiedPair(
    const cv::Mat& rectifiedLeft,
    const cv::Mat& rectifiedRight,
    int width,
    int height)
{
    cv::Mat concatenated;
    cv::hconcat(rectifiedLeft, rectifiedRight, concatenated);

    // optional lines
    for(int row=0; row<concatenated.rows; row+=64){
        cv::line(concatenated,
                 cv::Point(0,row),
                 cv::Point(concatenated.cols, row),
                 cv::Scalar(0,255,0), 2);
    }

    // Show or save
    // Example: Save to disk
    cv::imwrite("rectified_pair.jpg", concatenated);
}

/**
 * @brief Compute a disparity map using StereoBM
 */
inline cv::Mat computeDisparityMap(const cv::Mat& leftImage,
                                   const cv::Mat& rightImage)
{
    int numDisparities = 16; // adjust
    int blockSize = 9;       // adjust

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);
    cv::Mat disparity;
    stereoBM->compute(leftImage, rightImage, disparity);
    return disparity;
}

/**
 * @brief Convert disparity to depth (Z = f * B / disparity)
 */
inline cv::Mat computeDepthMap(const cv::Mat& disparity,
                               double focalLength,
                               double baseline)
{
    cv::Mat depthMap(disparity.size(), CV_64F);
    for(int y=0; y<disparity.rows; y++){
        for(int x=0; x<disparity.cols; x++){
            float d = disparity.at<float>(y,x);
            if(d>0.0f) {
                double Z = (focalLength * baseline)/ d;
                depthMap.at<double>(y,x) = Z;
            } else {
                depthMap.at<double>(y,x) = 0.0;
            }
        }
    }
    return depthMap;
}
