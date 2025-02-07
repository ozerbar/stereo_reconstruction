#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

/* =============================
 * 1) Utility functions: fundamental matrix, RANSAC, triangulation, etc.
 * =============================*/

/**
 * @brief Normalize 2D points and return the transformation matrix T
 */
void normalizePoints(const std::vector<cv::Point2f> &points,
                     std::vector<cv::Point2f> &normalizedPoints,
                     cv::Mat &T)
{
    // 1) Compute centroid
    cv::Point2f center(0, 0);
    for (auto &p : points)
        center += p;
    center *= (1.0f / points.size());

    // 2) Compute average distance
    double avgDist = 0.0;
    for (auto &p : points)
        avgDist += cv::norm(p - center);
    avgDist /= points.size();

    // 3) Scale factor
    double scale = std::sqrt(2.0) / avgDist;

    // 4) Construct a 3x3 transform matrix
    T = (cv::Mat_<double>(3,3) << 
         scale,    0,   -scale*center.x,
            0,   scale, -scale*center.y,
            0,     0,       1);

    // 5) Normalize the points with T
    normalizedPoints.resize(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::Mat p = (cv::Mat_<double>(3,1) << points[i].x, points[i].y, 1.0);
        cv::Mat p_norm = T * p;
        double w = p_norm.at<double>(2);
        normalizedPoints[i].x = static_cast<float>(p_norm.at<double>(0) / w);
        normalizedPoints[i].y = static_cast<float>(p_norm.at<double>(1) / w);
    }
}

/**
 * @brief Estimate the fundamental matrix F using the 8-point method + normalization
 */
cv::Mat computeFundamentalMatrix(const std::vector<cv::Point2f> &ptsLeft,
                                 const std::vector<cv::Point2f> &ptsRight)
{
    if (ptsLeft.size() < 8) {
        std::cerr << "[computeFundamentalMatrix] Not enough points!\n";
        return cv::Mat();
    }

    // 1) Normalize
    std::vector<cv::Point2f> normLeft, normRight;
    cv::Mat T1, T2;
    normalizePoints(ptsLeft,  normLeft,  T1);
    normalizePoints(ptsRight, normRight, T2);

    // 2) Construct matrix A
    size_t N = normLeft.size();
    Eigen::MatrixXd A(N, 9);
    for (size_t i = 0; i < N; i++)
    {
        double x1 = normLeft[i].x;
        double y1 = normLeft[i].y;
        double x2 = normRight[i].x;
        double y2 = normRight[i].y;

        A(i,0) = x2*x1;   A(i,1) = x2*y1;   A(i,2) = x2;
        A(i,3) = y2*x1;   A(i,4) = y2*y1;   A(i,5) = y2;
        A(i,6) = x1;      A(i,7) = y1;      A(i,8) = 1.0;
    }

    // 3) SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd f = V.col(8); // last column

    // 4) Convert to OpenCV Mat
    cv::Mat F(3,3, CV_64F);
    for (int i = 0; i < 9; i++) {
        F.at<double>(i/3, i%3) = f(i);
    }

    // 5) Enforce rank=2
    cv::SVD svdF(F);
    cv::Mat U = svdF.u, W = svdF.w, Vt = svdF.vt;
    W.at<double>(2,0) = 0.0;
    F = U * cv::Mat::diag(W) * Vt;

    // 6) Denormalize
    F = T2.t() * F * T1;
    return F;
}

/**
 * @brief Compute epipolar distance
 */
double computeEpipolarError(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F)
{
    cv::Mat x1 = (cv::Mat_<double>(3,1) << p1.x, p1.y, 1.0);
    cv::Mat line = F * x1; // [A B C]^T
    double A = line.at<double>(0,0);
    double B = line.at<double>(1,0);
    double C = line.at<double>(2,0);
    double denom = std::sqrt(A*A + B*B);
    if (denom < 1e-12) return 1e12; // avoid div0
    double dist = std::fabs(A*p2.x + B*p2.y + C)/denom;
    return dist;
}

/**
 * @brief Simple RANSAC for F estimation
 */
cv::Mat ransacFundamentalMatrix(std::vector<cv::Point2f> &pts1,
                                std::vector<cv::Point2f> &pts2,
                                int maxIters, float threshold,
                                int &bestInlierCount)
{
    bestInlierCount = 0;
    cv::Mat bestF;
    srand(static_cast<unsigned>(time(0)));

    if (pts1.size() < 8) {
        std::cerr << "[ransacFundamentalMatrix] Not enough points.\n";
        return cv::Mat();
    }

    for (int i = 0; i < maxIters; i++)
    {
        // randomly sample 8 points
        std::vector<cv::Point2f> s1, s2;
        for (int j = 0; j < 8; j++) {
            int idx = rand() % pts1.size();
            s1.push_back(pts1[idx]);
            s2.push_back(pts2[idx]);
        }
        cv::Mat F = computeFundamentalMatrix(s1, s2);
        if (F.empty()) continue;

        // count inliers
        int count = 0;
        for (size_t k = 0; k < pts1.size(); k++)
        {
            double err = computeEpipolarError(pts1[k], pts2[k], F);
            if (err < threshold) count++;
        }

        if (count > bestInlierCount) {
            bestInlierCount = count;
            bestF = F.clone();
        }
    }

    return bestF;
}

/* ========== Helper functions for E decomposition ========== */

/**
 * @brief Check if a 3D point is in front of the camera
 */
bool isInFrontOfCamera(const cv::Mat& P, const cv::Mat& point3D)
{
    cv::Mat proj = P * point3D;  // 3x1
    double w = proj.at<double>(2,0);
    double w3 = point3D.at<double>(3,0);
    return (w > 0.0 && w3 != 0.0);
}

/**
 * @brief Given P1, P2 and corresponding pixel points p1, p2, perform triangulation to get a 3D point (4x1)
 */
cv::Mat triangulatePoint(const cv::Mat& P1, const cv::Mat& P2,
                         const cv::Point2f& p1, const cv::Point2f& p2)
{
    cv::Mat A(4,4, CV_64F);
    A.row(0) = p1.x * P1.row(2) - P1.row(0);
    A.row(1) = p1.y * P1.row(2) - P1.row(1);
    A.row(2) = p2.x * P2.row(2) - P2.row(0);
    A.row(3) = p2.y * P2.row(2) - P2.row(1);

    cv::SVD svd(A);
    cv::Mat X = svd.vt.row(3).t(); // 4x1
    double w = X.at<double>(3,0);
    if (std::fabs(w) < 1e-12) w = 1e-12;
    X /= w;
    return X;
}

/**
 * @brief Decompose the essential matrix E to find the best (R, t)
 */
void decomposeEssentialMatrix(const cv::Mat& E,
                              const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& K1,
                              cv::Mat& bestR,
                              cv::Mat& bestT)
{
    // ... omitted for brevity ...
}

/* ==========  SGBM disparity function ========== */
cv::Mat computeDisparitySGBM(const cv::Mat &leftGray, const cv::Mat &rightGray,
                             int numDisparities=64, int blockSize=11)
{
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0,                 // minDisparity
        numDisparities,    // numDisparities
        blockSize
    );
    // Set additional parameters here if needed

    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity);
    return disparity;
}


/* =============================
 * 2) Main: produce three disparities
 * =============================*/

int main()
{
    // 1) Load images
    std::string leftImagePath  = "../Datasets/Shopvac-imperfect/im0.png";
    std::string rightImagePath = "../Datasets/Shopvac-imperfect/im1.png";

    cv::Mat leftImage  = cv::imread(leftImagePath,  cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_COLOR);
    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << "[Error] Failed to load images.\n";
        return -1;
    }

    // 2) Define feature methods
    struct FeatureMethod {
        std::string name;
        cv::Ptr<cv::Feature2D> extractor;
        int normType;
    };
    std::vector<FeatureMethod> methods = {
        { "ORB",   cv::ORB::create(),   cv::NORM_HAMMING },
        { "SIFT",  cv::SIFT::create(),  cv::NORM_L2      },
        { "BRISK", cv::BRISK::create(), cv::NORM_HAMMING }
    };

    // 3) Camera intrinsics
    cv::Mat K0 = (cv::Mat_<double>(3,3) << 7228.4,  0,      1112.085,
                                           0,       7228.4, 1010.431,
                                           0,       0,      1.0 );
    cv::Mat K1 = (cv::Mat_<double>(3,3) << 7228.4,  0,      1628.613,
                                           0,       7228.4, 1010.431,
                                           0,       0,      1.0 );
    cv::Mat D0 = cv::Mat::zeros(1,5, CV_64F);
    cv::Mat D1 = cv::Mat::zeros(1,5, CV_64F);

    // Assume some extrinsics; for demonstration, R=I, T in X
    double baseline = 379.965;  // example
    cv::Mat R = cv::Mat::eye(3,3, CV_64F);
    // negative so left camera is "to the left" of right
    cv::Mat T = (cv::Mat_<double>(3,1) << -baseline, 0, 0);

    // 4) Loop over each method
    for (auto &method : methods)
    {
        std::cout << "\n========== " << method.name << " ==========\n";

        // (a) Detect and compute descriptors
        cv::Mat leftGray, rightGray;
        cv::cvtColor(leftImage,  leftGray,  cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kpLeft, kpRight;
        cv::Mat descLeft, descRight;
        method.extractor->detectAndCompute(leftGray,  cv::noArray(), kpLeft,  descLeft);
        method.extractor->detectAndCompute(rightGray, cv::noArray(), kpRight, descRight);
        if (descLeft.empty() || descRight.empty()) {
            std::cerr << "[WARN] " << method.name << ": no descriptors.\n";
            continue;
        }

        // (b) KNN match + ratio test
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(method.normType, false);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(descLeft, descRight, knn, 2);

        std::vector<cv::DMatch> goodMatches;
        double ratioThresh = 0.75;
        for (auto &m : knn) {
            if (m.size() == 2) {
                float dist1 = m[0].distance;
                float dist2 = m[1].distance;
                if (dist1 < ratioThresh * dist2) {
                    goodMatches.push_back(m[0]);
                }
            }
        }

        if (goodMatches.size() < 8) {
            std::cerr << "[WARN] " << method.name << ": Not enough good matches.\n";
            continue;
        }

        // (c) Collect matched points
        std::vector<cv::Point2f> ptsLeft, ptsRight;
        ptsLeft.reserve(goodMatches.size());
        ptsRight.reserve(goodMatches.size());
        for (auto &gm : goodMatches) {
            ptsLeft.push_back( kpLeft[gm.queryIdx].pt );
            ptsRight.push_back(kpRight[gm.trainIdx].pt );
        }

        // (d) RANSAC => Fundamental matrix
        int inlierCount = 0;
        cv::Mat F = ransacFundamentalMatrix(ptsLeft, ptsRight, 2000, 1.5f, inlierCount);
        if (F.empty()) {
            std::cerr << "[WARN] " << method.name << ": F is empty.\n";
            continue;
        }
        std::cout << "[INFO] " << method.name << " => #inliers=" << inlierCount << "\n";

        /************************************************************
         * (1) Uncalibrated Rectification + Disparity
         ************************************************************/
        {
            // stereoRectifyUncalibrated => H1, H2
            cv::Mat H1, H2;
            cv::stereoRectifyUncalibrated(ptsLeft, ptsRight, F, leftImage.size(), H1, H2);

            // Warp perspective
            cv::Mat rectLeftU, rectRightU;
            cv::warpPerspective(leftImage,  rectLeftU,  H1, leftImage.size());
            cv::warpPerspective(rightImage, rectRightU, H2, leftImage.size());

            // Compute disparity from the uncalibrated rectified pair
            cv::Mat rectLeftUGray, rectRightUGray;
            cv::cvtColor(rectLeftU,  rectLeftUGray,  cv::COLOR_BGR2GRAY);
            cv::cvtColor(rectRightU, rectRightUGray, cv::COLOR_BGR2GRAY);

            // (Optional) apply CLAHE
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
            clahe->apply(rectLeftUGray,  rectLeftUGray);
            clahe->apply(rectRightUGray, rectRightUGray);

            cv::Mat dispUncalib = computeDisparitySGBM(rectLeftUGray, rectRightUGray);

            double minv, maxv;
            cv::minMaxLoc(dispUncalib, &minv, &maxv);
            cv::Mat dispU8;
            dispUncalib.convertTo(dispU8, CV_8U, 255.0/(maxv - minv + 1e-6));

            // Save
            std::string dispUName = "uncalib_disparity_" + method.name + ".png";
            cv::imwrite(dispUName, dispU8);
        }

        /************************************************************
         * (2) Calibrated Rectification (alpha=0) => cropped disparity
         ************************************************************/
        {
            cv::Mat R1, R2, P1, P2, Q;
            cv::Rect validRoi1, validRoi2;
            // alpha=0 => zoomed so only valid region is used
            cv::stereoRectify(K0, D0, K1, D1, 
                              leftImage.size(), R, T,
                              R1, R2, P1, P2, Q,
                              /*flags=*/0, /*alpha=*/0, 
                              leftImage.size(), &validRoi1, &validRoi2);

            // initUndistortRectifyMap
            cv::Mat map1x, map1y, map2x, map2y;
            cv::initUndistortRectifyMap(K0, D0, R1, P1, leftImage.size(), CV_32FC1, map1x, map1y);
            cv::initUndistortRectifyMap(K1, D1, R2, P2, rightImage.size(), CV_32FC1, map2x, map2y);

            // Remap
            cv::Mat rectLeft0, rectRight0;
            cv::remap(leftImage,  rectLeft0,  map1x, map1y, cv::INTER_LINEAR);
            cv::remap(rightImage, rectRight0, map2x, map2y, cv::INTER_LINEAR);

            // Convert to gray
            cv::Mat left0Gray, right0Gray;
            cv::cvtColor(rectLeft0,  left0Gray,  cv::COLOR_BGR2GRAY);
            cv::cvtColor(rectRight0, right0Gray, cv::COLOR_BGR2GRAY);

            // Disparity
            // e.g. use bigger numDisparities if needed
            cv::Mat dispC0 = computeDisparitySGBM(left0Gray, right0Gray, 128, 11);

            double minVal, maxVal;
            cv::minMaxLoc(dispC0, &minVal, &maxVal);
            cv::Mat dispC0_8U;
            dispC0.convertTo(dispC0_8U, CV_8U, 255.0/(maxVal - minVal + 1e-6));

            // Save
            std::string dispC0_Name = "calib_disparity_alpha0_" + method.name + ".png";
            cv::imwrite(dispC0_Name, dispC0_8U);
        }

        /************************************************************
         * (3) Calibrated Rectification (alpha=1) => keep all pixels
         ************************************************************/
        {
            cv::Mat R1, R2, P1, P2, Q;
            cv::Rect validRoi1, validRoi2;
            // alpha=1 => keep as much as possible
            cv::stereoRectify(K0, D0, K1, D1, 
                              leftImage.size(), R, T,
                              R1, R2, P1, P2, Q,
                              /*flags=*/0, /*alpha=*/1, 
                              leftImage.size(), &validRoi1, &validRoi2);

            // initUndistortRectifyMap
            cv::Mat map1x, map1y, map2x, map2y;
            cv::initUndistortRectifyMap(K0, D0, R1, P1, leftImage.size(), CV_32FC1, map1x, map1y);
            cv::initUndistortRectifyMap(K1, D1, R2, P2, rightImage.size(), CV_32FC1, map2x, map2y);

            // Remap
            cv::Mat rectLeft1, rectRight1;
            cv::remap(leftImage,  rectLeft1,  map1x, map1y, cv::INTER_LINEAR);
            cv::remap(rightImage, rectRight1, map2x, map2y, cv::INTER_LINEAR);

            // Convert to gray
            cv::Mat left1Gray, right1Gray;
            cv::cvtColor(rectLeft1,  left1Gray,  cv::COLOR_BGR2GRAY);
            cv::cvtColor(rectRight1, right1Gray, cv::COLOR_BGR2GRAY);

            // Disparity
            cv::Mat dispC1 = computeDisparitySGBM(left1Gray, right1Gray, 128, 11);

            double minVal, maxVal;
            cv::minMaxLoc(dispC1, &minVal, &maxVal);
            cv::Mat dispC1_8U;
            dispC1.convertTo(dispC1_8U, CV_8U, 255.0/(maxVal - minVal + 1e-6));

            // Save
            std::string dispC1_Name = "calib_disparity_alpha1_" + method.name + ".png";
            cv::imwrite(dispC1_Name, dispC1_8U);
        }

    } // end for(methods)

    std::cout << "\nDone. For each method, you should have:\n"
              << " (1) uncalib_disparity_<method>.png\n"
              << " (2) calib_disparity_alpha0_<method>.png\n"
              << " (3) calib_disparity_alpha1_<method>.png\n\n";
    return 0;
}
