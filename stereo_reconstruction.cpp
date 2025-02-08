#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <cmath>
#include <opencv2/ximgproc.hpp>
#include <fstream>

#include "depth_map_generation.h"
#include "mesh_generation.h"


void normalizePoints(const std::vector<cv::Point2f>& points, 
                    std::vector<cv::Point2f>& normalizedPoints,
                    cv::Mat& T) {
    // Compute centroid
    cv::Point2f center(0,0);
    for (const auto& p : points) {
        center += p;
    }
    center *= (1.0 / points.size());

    // Compute average distance
    double avgDist = 0;
    for (const auto& p : points) {
        avgDist += cv::norm(p - center);
    }
    avgDist /= points.size();

    // Scale factor
    double scaleFactor = std::sqrt(2.0) / avgDist;

    // Construct the transformation matrix
    T = (cv::Mat_<double>(3, 3) << scaleFactor, 0, -scaleFactor * center.x,
                                   0, scaleFactor, -scaleFactor * center.y,
                                   0, 0, 1);

    // Normalize the points
    normalizedPoints.resize(points.size());
    for (size_t i=0; i<points.size(); i++) {
        cv::Mat p = (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, 1.0);
        cv::Mat p_norm = T * p; 
        normalizedPoints[i].x = p_norm.at<double>(0)/p_norm.at<double>(2);
        normalizedPoints[i].y = p_norm.at<double>(1)/p_norm.at<double>(2);
    }
}

/**
 * @brief Estimate the fundamental matrix F using the 8-point algorithm or an overdetermined linear method
 * @param pointsLeft   Points from the left image
 * @param pointsRight  Points from the right image
 * @note  If more than 8 points are provided, a least-squares solution via SVD is applied to all points
 */
cv::Mat computeFundamentalMatrix(const std::vector<cv::Point2f> &pointsLeft,
                                 const std::vector<cv::Point2f> &pointsRight)
{
    // Need at least 8 pairs of points
    size_t N = pointsLeft.size();
    if (N < 8)
    {
        std::cerr << "[computeFundamentalMatrix] Not enough points!" << std::endl;
        return cv::Mat();
    }

    // NEW: Normalize points for numeric stability
    std::vector<cv::Point2f> normPointsLeft, normPointsRight;
    cv::Mat T1, T2;
    normalizePoints(pointsLeft, normPointsLeft, T1);
    normalizePoints(pointsRight, normPointsRight, T2);


    // Construct an N x 9 matrix A
    Eigen::MatrixXd A(N, 9);
    for (size_t i = 0; i < N; ++i)
    {
        double x1 = normPointsLeft[i].x;  
        double y1 = normPointsLeft[i].y;
        double x2 = normPointsRight[i].x;
        double y2 = normPointsRight[i].y;

        A(i, 0) = x2 * x1;
        A(i, 1) = x2 * y1;
        A(i, 2) = x2;
        A(i, 3) = y2 * x1;
        A(i, 4) = y2 * y1;
        A(i, 5) = y2;
        A(i, 6) = x1;
        A(i, 7) = y1;
        A(i, 8) = 1.0;
    }

    // Use ComputeFullU | ComputeFullV to avoid degenerate issues
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    // If the number of columns is < 9 (degenerate), skip it
    if (V.cols() < 9)
    {
        std::cerr << "[computeFundamentalMatrix] WARNING: SVD degenerate, V.cols()="
                  << V.cols() << " < 9. Skip.\n";
        return cv::Mat();
    }

    // The last column of V (index 8) is the eigenvector corresponding to the smallest singular value (9x1)
    Eigen::VectorXd f = V.col(8);

    // Put the 9 elements of f into a 3x3 matrix
    cv::Mat F(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i)
    {
        F.at<double>(i / 3, i % 3) = f(i);
    }

    // Enforce rank-2: do an SVD on F and set the third singular value to zero
    cv::SVD svdF(F);
    cv::Mat U = svdF.u;
    cv::Mat W = svdF.w;  // 3x1
    cv::Mat Vt = svdF.vt;

    W.at<double>(2, 0) = 0.0; // Force rank = 2
    F = U * cv::Mat::diag(W) * Vt;

    // NEW: Denormalize the fundamental matrix
    F = T2.t() * F * T1;

    return F;
}

/**
 * @brief Compute epipolar reprojection error: x2^T * F * x1
 * @param pointsLeft   Points from the left image
 * @param pointsRight  Points from the right image
 * @param F            The fundamental matrix
 * @return             Returns the sum of squared errors
 */
double computeReprojectionError(
    const std::vector<cv::Point2f> &pointsLeft,
    const std::vector<cv::Point2f> &pointsRight,
    const cv::Mat &F)
{
    if (pointsLeft.size() != pointsRight.size())
    {
        std::cerr << "[computeReprojectionError] Mismatch in points size!" << std::endl;
        return -1.0;
    }

    std::vector<double> errors;
    errors.reserve(pointsLeft.size());

    for (size_t i = 0; i < pointsLeft.size(); ++i)
    {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pointsLeft[i].x, pointsLeft[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3, 1) << pointsRight[i].x, pointsRight[i].y, 1.0);

        // The epipolar constraint error: | x2^T * F * x1 |
        double error = std::abs(x2.dot(F * x1));
        errors.push_back(error);
    }

    // Here we simply return the sum of squares
    double sum_sq = 0.0;
    for (auto &e : errors)
        sum_sq += (e * e);

    return sum_sq;
}

double computeEpipolarError(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &F)
{
    // Create a homogeneous coordinate for p1 (in image 1)
    cv::Mat x1 = (cv::Mat_<double>(3, 1) << p1.x, p1.y, 1.0);

    // Calculate the epipolar line in the second image using F
    cv::Mat line = F * x1; // Line = F * x1 (Epipolar line in the second image)

    // The line equation is Ax + By + C = 0, where line = [A B C]
    double A = line.at<double>(0, 0);
    double B = line.at<double>(1, 0);
    double C = line.at<double>(2, 0);

    // Compute the error between point p2 and the epipolar line
    double denom = A * A + B * B;
    double sqrt_res = std::sqrt(denom);
    double error = std::abs(A * p2.x + B * p2.y + C) / sqrt_res;

    return error;
}

/**
 * @brief Estimate the fundamental matrix using RANSAC to handle outliers
 * @param points1       Points from the first image
 * @param points2       Points from the second image
 * @param maxIterations Maximum number of RANSAC iterations
 * @param threshold     Epipolar error threshold
 * @param inlierCount   Reference to an integer that will store the count of inliers
 * @return              Returns the best estimated fundamental matrix (3x3)
 */
cv::Mat ransacFundamentalMatrix(std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, int maxIterations, float threshold, int& inlierCount) {
    int bestInlierCount = 0;
    cv::Mat bestF;
    std::vector<cv::Point2f> bestInliers1, bestInliers2;

    srand(time(0));  // Seed the random number generator

   
    for (int i = 0; i < maxIterations; ++i) {
        
        std::vector<cv::Point2f> samplePoints1, samplePoints2;
        for (int j = 0; j < 8; ++j) {
            int idx = rand() % points1.size(); 
            samplePoints1.push_back(points1[idx]);
            samplePoints2.push_back(points2[idx]);
        }

       
        cv::Mat F = computeFundamentalMatrix(samplePoints1, samplePoints2);

       
        std::vector<cv::Point2f> inliers1, inliers2;
        int count = 0;
        for (size_t j = 0; j < points1.size(); ++j) {
            double error = computeEpipolarError(points1[j], points2[j], F);
            if (error < threshold) {
                inliers1.push_back(points1[j]);
                inliers2.push_back(points2[j]);
                count++;
            }
        }

    
        if (count > bestInlierCount) {
            bestInlierCount = count;
            bestF = F.clone();
            bestInliers1 = inliers1;
            bestInliers2 = inliers2;
        }
    }

    inlierCount = bestInlierCount;
    return bestF;  
}

/**
 * @brief Compute disparity map from two rectified images using Block Matching
 * @param leftImage   rectified left image
 * @param rightImage  rectified right image
 * @return            disparity map
 */
cv::Mat computeDisparityMap(const cv::Mat& leftImage, const cv::Mat& rightImage)
{
    int numDisparities = 16;  
    int blockSize = 9;        

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);

    cv::Mat disparity; 
    stereoBM->compute(leftImage, rightImage, disparity);

    return disparity;
}

/**
 * @brief Convert disparity map to depth map using camera parameters
 * @param disparity    The disparity map
 * @param focalLength  The focal length (fx)
 * @param baseline     The baseline
 * @return             depth map
 */
cv::Mat computeDepthMap(const cv::Mat& disparity, double focalLength, double baseline)
{
    cv::Mat depthMap(disparity.size(), CV_64F);
    
    // First, normalize disparity values to ensure they're in a good range
    cv::Mat normalizedDisparity;
    cv::normalize(disparity, normalizedDisparity, 0, 255, cv::NORM_MINMAX, CV_32F);
    
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            double d = normalizedDisparity.at<float>(y, x);
            if (d > 0) {
                // Add a small epsilon to avoid division by zero
                double depth = (focalLength * baseline) / (d + 1e-10);
                depthMap.at<double>(y, x) = depth;
            } else {
                depthMap.at<double>(y, x) = 0.0;
            }
        }
    }
    return depthMap;
}


/**
 * @brief Check if a 3D point is in front of a camera
 * @param P       The camera projection matrix
 * @param point3D 3D point in homogeneous coordinates
 * @return        True if the point is in front of the camera
 */
bool isInFrontOfCamera(const cv::Mat& P, const cv::Mat& point3D) {
    cv::Mat homogeneous_point;
    if (point3D.rows == 3)
        homogeneous_point = cv::Mat::ones(4, 1, CV_64F);
    else
        homogeneous_point = point3D.clone();
    
    if (point3D.rows == 3) {
        point3D.copyTo(homogeneous_point.rowRange(0, 3));
    }

    cv::Mat projected = P * homogeneous_point;
    double depth = projected.at<double>(2) / projected.at<double>(3);
    
    return depth > 0;
}

/**
 * @brief Triangulate a 3D point from two 2D points and two projection matrices
 * @param P1   Projection matrix of the first camera
 * @param P2   Projection matrix of the second camera
 * @param pt1  First 2D point
 * @param pt2  Second 2D point
 * @return     3D point in homogeneous coordinates
 */
cv::Mat triangulatePoint(const cv::Mat& P1, const cv::Mat& P2, 
                        const cv::Point2f& pt1, const cv::Point2f& pt2) {
    cv::Mat A(4, 4, CV_64F);
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);
    
    cv::SVD svd(A);
    cv::Mat point3D = svd.vt.row(3).t();
    point3D = point3D / point3D.at<double>(3);
    
    return point3D;
}

/**
 * @brief Decompose the essential matrix into rotation and translation
 * @param E         The essential matrix
 * @param points1   Points from the first image
 * @param points2   Points from the second image
 * @param best_R    The best rotation matrix
 * @param best_t    The best translation vector
 * @note This function tries all possible combinations of R and T
 */
void decomposeEssentialMatrix(const cv::Mat& E, 
                            const std::vector<cv::Point2f>& points1,
                            const std::vector<cv::Point2f>& points2,
                            cv::Mat& best_R, 
                            cv::Mat& best_t) {
    cv::SVD svd(E);
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
                                          1, 0, 0,
                                          0, 0, 1);

    std::vector<cv::Mat> R_possibilities{
        svd.u * W * svd.vt,
        svd.u * W.t() * svd.vt,
    };
    std::vector<cv::Mat> t_possibilities{
        svd.u.col(2),
        -svd.u.col(2)
    };

    for (auto& R : R_possibilities) {
        if (cv::determinant(R) < 0) {
            R = -R;
        }
    }

    cv::Mat P1 = cv::Mat::eye(4, 4, CV_64F);
    P1 = P1(cv::Rect(0, 0, 4, 3));

    int max_points_in_front = 0;
    int best_R_idx = 0;
    int best_t_idx = 0;

    for (size_t r = 0; r < R_possibilities.size(); ++r) {
        for (size_t t = 0; t < t_possibilities.size(); ++t) {
            cv::Mat P2 = cv::Mat::eye(4, 4, CV_64F);
            R_possibilities[r].copyTo(P2(cv::Rect(0, 0, 3, 3)));
            t_possibilities[t].copyTo(P2(cv::Rect(3, 0, 1, 3)));
            P2 = P2(cv::Rect(0, 0, 4, 3));

            int points_in_front = 0;
            
            for (size_t i = 0; i < points1.size(); ++i) {
                cv::Mat point3D = triangulatePoint(P1, P2, points1[i], points2[i]);
                if (isInFrontOfCamera(P1, point3D) && 
                    isInFrontOfCamera(P2, point3D)) {
                    points_in_front++;
                }
            }

            if (points_in_front > max_points_in_front) {
                max_points_in_front = points_in_front;
                best_R_idx = r;
                best_t_idx = t;
            }
        }
    }

    best_R = R_possibilities[best_R_idx].clone();
    best_t = t_possibilities[best_t_idx].clone();

    std::cout << "Selected solution has " << max_points_in_front
              << " points in front of both cameras out of "
              << points1.size() << " total points." << std::endl;
}

/**
 * @brief Perform stereo rectification and compute disparity map from stereo images
 * @param F           The fundamental matrix
 * @param K1          The intrinsic matrix of the left camera
 * @param K2          The intrinsic matrix of the right camera
 * @param leftImage   The left input image
 * @param rightImage  The right input image
 * @note              This function first computes the essential matrix from F and K1, K2,
 *                    then uses SVD to extract the rotation and translation. After rectifying,
 *                    it computes disparity using block matching.
 */
cv::Mat stereoRectifyAndComputeDisparity(const cv::Mat& F, const cv::Mat& K1, const cv::Mat& K2,
                                      const cv::Mat& leftImage, const cv::Mat& rightImage,
                                      const std::vector<cv::Point2f>& pointsLeft,
                                      const std::vector<cv::Point2f>& pointsRight, 
                                      std::string path)
{
    cv::Mat E = K2.t() * F * K1;
    std::cout << "Essential matrix E:\n" << E << std::endl;

    cv::Mat T, R;
    decomposeEssentialMatrix(E, pointsLeft, pointsRight, R, T);

    std::cout << "Rotation matrix R:\n" << R << std::endl;
    std::cout << "Translation vector T:\n" << T << std::endl;

    CV_Assert(!K1.empty() && !K2.empty() && K1.size() == cv::Size(3,3) && K2.size() == cv::Size(3,3));
    CV_Assert(!leftImage.empty() && !rightImage.empty());

    std::cout << "Debug Image size: " << leftImage.size() << std::endl;
    std::cout << "Debug K1: " << K1.type() << " size: " << K1.size() << std::endl;
    std::cout << "Debug K2: " << K2.type() << " size: " << K2.size() << std::endl;

    cv::Mat R1, R2, P1, P2, Q;
    try {
        cv::stereoRectify(K1, cv::Mat(), K2, cv::Mat(), leftImage.size(), R, T,
                          R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1.0);

        std::cout << "Debug R1: " << R1.size() << " type: " << R1.type() << std::endl;
        std::cout << "Debug P1: " << P1.size() << " type: " << P1.type() << std::endl;

        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(K1, cv::Mat(), R1, P1,
                                    leftImage.size(), CV_32FC1, map1x, map1y); 
        cv::initUndistortRectifyMap(K2, cv::Mat(), R2, P2,
                                    rightImage.size(), CV_32FC1, map2x, map2y);

        cv::Mat rectifiedLeft, rectifiedRight;
        cv::remap(leftImage, rectifiedLeft, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(rightImage, rectifiedRight, map2x, map2y, cv::INTER_LINEAR);

        cv::Mat disparity = computeDisparityMap(rectifiedLeft, rectifiedRight);

        // Create stitched image
        cv::Mat stitched(
            rectifiedLeft.rows,
            rectifiedLeft.cols + rectifiedRight.cols,
            rectifiedLeft.type()
        );
        rectifiedLeft.copyTo(
            stitched(cv::Rect(0, 0, rectifiedLeft.cols, rectifiedLeft.rows))
        );
        rectifiedRight.copyTo(
            stitched(cv::Rect(rectifiedLeft.cols, 0, rectifiedRight.cols, rectifiedRight.rows))
        );

        for (size_t i = 0; i < pointsLeft.size(); ++i)
        {
            float xL = pointsLeft[i].x;
            float yL = pointsLeft[i].y;
            if (xL < 0 || xL >= map1x.cols || yL < 0 || yL >= map1x.rows) continue;

            float rectLx = map1x.at<float>((int) yL, (int) xL);
            float rectLy = map1y.at<float>((int) yL, (int) xL);

            float xR = pointsRight[i].x;
            float yR = pointsRight[i].y;
            if (xR < 0 || xR >= map2x.cols || yR < 0 || yR >= map2x.rows) continue;

            float rectRx = map2x.at<float>((int) yR, (int) xR);
            float rectRy = map2y.at<float>((int) yR, (int) xR);

            rectRx += static_cast<float>(rectifiedLeft.cols);

            cv::circle(stitched, cv::Point2f(rectLx, rectLy), 4, cv::Scalar(0, 255, 0), -1);
            cv::circle(stitched, cv::Point2f(rectRx, rectRy), 4, cv::Scalar(0, 255, 0), -1);

            cv::line(
                stitched,
                cv::Point2f(rectLx, rectLy),
                cv::Point2f(rectRx, rectRy),
                cv::Scalar(255, 0, 0),
                2
            );
        }
        cv::imwrite(path + "_rectified_stitched_matches.jpg", stitched);

        cv::imwrite(path + "_rectified_left.jpg", rectifiedLeft);
        cv::imwrite(path + "_rectified_right.jpg", rectifiedRight);
        cv::imwrite(path + "_disparity_map.jpg", disparity);
        std::cout<<"disparity map has outputed successfully"<<std::endl;
        return disparity;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return cv::Mat();
}


cv::Mat createDepthVis(cv::Mat depthMap) {
    // Find valid depth range
    double minVal, maxVal;
    cv::minMaxLoc(depthMap, &minVal, &maxVal, nullptr, nullptr, depthMap > 0);

    // Remove outliers (values too far from median)
    cv::Mat mask = depthMap > 0;
    std::vector<float> validDepths;
    for(int y = 0; y < depthMap.rows; y++) {
        for(int x = 0; x < depthMap.cols; x++) {
            if(mask.at<uint8_t>(y,x)) {
                validDepths.push_back(depthMap.at<float>(y,x));
            }
        }
    }

    std::sort(validDepths.begin(), validDepths.end());
    float Q1 = validDepths[validDepths.size() * 0.25];
    float Q3 = validDepths[validDepths.size() * 0.75];
    float IQR = Q3 - Q1;
    float lowerBound = Q1 - 1.5 * IQR;
    float upperBound = Q3 + 1.5 * IQR;

    // Create visualization with outlier removal
    cv::Mat normalized;
    depthMap.copyTo(normalized);
    normalized.setTo(0, depthMap < lowerBound);
    normalized.setTo(0, depthMap > upperBound);

    // Normalize valid depths
    cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1, normalized > 0);

    // Apply color mapping
    cv::Mat colorized;
    cv::applyColorMap(normalized, colorized, cv::COLORMAP_VIRIDIS); // Try VIRIDIS for better depth perception

    // Make invalid points black
    colorized.setTo(cv::Scalar(0,0,0), normalized == 0);

    // Optional: Apply slight Gaussian blur for smoother visualization
    cv::GaussianBlur(colorized, colorized, cv::Size(3,3), 0);

    return colorized;
}


cv::Mat computeDepthMap2(const cv::Mat& disparity, double focalLength, double baseline) {
    cv::Mat depthMap(disparity.size(), CV_64F);

    // First pass: compute raw depth values
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            float d = disparity.at<float>(y, x);
            if (d > 0) {
                depthMap.at<double>(y, x) = (focalLength * baseline) / d;
            } else {
                depthMap.at<double>(y, x) = 0.0;
            }
        }
    }

    // Filter out noise and fill small holes
    cv::Mat filteredDepth;

    // Apply bilateral filter to reduce noise while preserving edges
    // cv::Mat temp;
    // depthMap.convertTo(temp, CV_32F);
    // cv::bilateralFilter(temp, filteredDepth, 7, 50, 50);

    // Fill small holes using morphological operations
    cv::Mat mask = (filteredDepth > 0);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // Inpaint remaining small holes
    cv::Mat inpainted;
    cv::inpaint(filteredDepth, ~mask, inpainted, 3, cv::INPAINT_TELEA);

    return createDepthVis(inpainted);
}




int pipeline() {
     std::string dataset = "artroom1";
    std::string path = "/workspace/Datasets/" + dataset + "/";
    cv::Mat leftImage = cv::imread(path + "im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat rightImage = cv::imread(path + "im1.png", cv::IMREAD_GRAYSCALE);
    if (leftImage.empty() || rightImage.empty())
    {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    double baseline = 0.53662;
    double focalLength = 1733.74;

    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2,
                                           cv::ORB::HARRIS_SCORE, 31, 20);

    orb->detect(leftImage, keypointsLeft);
    orb->detect(rightImage, keypointsRight);

    orb->compute(leftImage, keypointsLeft, descriptorsLeft);
    orb->compute(rightImage, keypointsRight, descriptorsRight);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    std::cout<<"the size of matches:"<<matches.size()<<std::endl;
    std::cout<<"the min dist:"<<min_dist<<std::endl;

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
    std::cout<<"the size of good_matches:"<<good_matches.size()<<std::endl;

    std::vector<cv::Point2f> ptsLeft, ptsRight;
    for (unsigned int i = 0; i < good_matches.size(); ++i)
    {
        ptsLeft.push_back(keypointsLeft[good_matches[i].queryIdx].pt);
        ptsRight.push_back(keypointsRight[good_matches[i].trainIdx].pt);
    }
    if (ptsLeft.size() < 8)
    {
        std::cerr << "Error: Not enough matches to compute Fundamental Matrix!" << std::endl;
        return -1;
    }

    cv::Mat F_all = computeFundamentalMatrix(ptsLeft, ptsRight);
    std::cout << "[Direct] F estimated from all points: \n"
              << F_all << std::endl;
    double error_all = computeReprojectionError(ptsLeft, ptsRight, F_all);
    std::cout << "[Direct] Reprojection error (sum of squares): " << error_all << std::endl;

    int maxIterations = 3000;
    float threshold = 1.3;
    int inlierCount = 0;

    cv::Mat F = ransacFundamentalMatrix(ptsLeft, ptsRight, maxIterations, threshold, inlierCount);
    std::cout << "[RANSAC] F estimated: "<< F << std::endl;
    double error_ransac = computeReprojectionError(ptsLeft, ptsRight, F);
    std::cout << "[RANSAC] Reprojection error (sum of squares): " << ptsLeft.size() << std::endl;
    std::cout << "The Number of total matched point pairs: " << inlierCount<< std::endl;
    std::cout << "The Number of Inlier: " << inlierCount<< std::endl;

    const int numMatchesToDraw = 10;
    std::vector<cv::DMatch> topMatches(matches.begin(),
                                       matches.begin() + std::min<int>(matches.size(), numMatchesToDraw));

    cv::Mat matchImage;
    cv::drawMatches(leftImage, keypointsLeft,
                    rightImage, keypointsRight,
                    topMatches, matchImage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    for (auto &m : topMatches)
    {
        cv::Point2f ptLeft  = keypointsLeft[m.queryIdx].pt;
        cv::Point2f ptRight = keypointsRight[m.trainIdx].pt;
        ptRight.x += (float)leftImage.cols;

        cv::line(matchImage, ptLeft, ptRight, cv::Scalar(0, 255, 0), 2);
    }

    std::string outputPath = "results/" + dataset + "_keypoints_with_opencv.jpg";
    cv::imwrite(outputPath, matchImage);
    // if (!cv::imwrite(outputPath, matchImage))
    // {
    //     std::cerr << "Error: Could not save the match image." << std::endl;
    //    return -1;
    // }
    std::cout << "Match image saved to " << outputPath << std::endl;

    cv::Mat cam0 = (cv::Mat_<double>(3, 3) <<
        1758.23, 0, 953.34,
        0, 1758.23, 552.29,
        0, 0, 1);

    cv::Mat cam1 = (cv::Mat_<double>(3, 3) <<
        1758.23, 0, 953.34,
        0, 1758.23, 552.29,
        0, 0, 1);

    std::cout << "Camera intrinsics for cam0:\n"
              << "fx=" << cam0.at<double>(0, 0) << ", fy=" << cam0.at<double>(1, 1)
              << ", cx=" << cam0.at<double>(0, 2) << ", cy=" << cam0.at<double>(1, 2) << std::endl;

    std::string resultPath = "results/" + dataset ;
    cv::Mat disparity = stereoRectifyAndComputeDisparity(F_all, cam0, cam1, leftImage, rightImage, ptsLeft, ptsRight, resultPath);

    // load disparity map from jpg image
    cv::Mat disparity_cor = imread("/workspace/disparity_image.jpg", cv::IMREAD_GRAYSCALE);

    // Compute and visualize depth map
    cv::Mat depthMap = computeDepthMap2(disparity_cor, 1733.74, 0.53662);

    // cv::Mat depthColor = createDepthVis(depthMap);


    /*
    // Normalize the depth map to ensure good visualization
    cv::Mat normalizedDepthMap;
    cv::normalize(depthMap, normalizedDepthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Apply bilateral filter to reduce noise while preserving edges
    cv::Mat filtered;
    cv::bilateralFilter(normalizedDepthMap, filtered, 9, 75, 75);

    // double minVal, maxVal;
    // cv::minMaxLoc(depthMap, &minVal, &maxVal);

    // cv::Mat depth8U;
    // depthMap.convertTo(depth8U, CV_8UC1, 255.0 / (maxVal + 1e-6));

    cv::Mat depthColor;
    cv::applyColorMap(filtered, depthColor, cv::COLORMAP_JET);
    */

    cv::imwrite(resultPath + "_depth_map_color.jpg", depthMap);
    std::cout << "Depth map successfully output " << std::endl;

    return 0;

}


int main()
{
    cv::Mat cam0 = (cv::Mat_<double>(3, 3) <<
        1758.23, 0, 953.34,
        0, 1758.23, 552.29,
        0, 0, 1);

    cv::Mat cam1 = (cv::Mat_<double>(3, 3) <<
        1758.23, 0, 953.34,
        0, 1758.23, 552.29,
        0, 0, 1);

    std::string dataset = "artroom1";
    float focal_length = 1733.74;
    float baseline = 536.62;

    std::string disparity_gt_path = "/workspace/Datasets/" + dataset + "/disp0.pfm";
    cv::Mat disparity_gt = readPFM(disparity_gt_path);

    // Print information about the disparity map
    std::cout << "Disparity map info:" << std::endl;
    std::cout << "Size: " << disparity_gt.size() << std::endl;
    std::cout << "Type: " << disparity_gt.type() << std::endl;

    // Check value range
    double minVal, maxVal;
    cv::minMaxLoc(disparity_gt, &minVal, &maxVal);
    std::cout << "[DISP] Min value: " << minVal << std::endl;
    std::cout << "[DISP] Max value: " << maxVal << std::endl;


    // --- Create visualization with proper (gray) scaling
    cv::Mat disparity_vis;
    std::string disparity_out_path = "/workspace/results/" + dataset + "_disparity_from_pfm.png";
    cv::minMaxLoc(disparity_gt, &minVal, &maxVal);
    disparity_gt.convertTo(disparity_vis, CV_32F, 1.0/(maxVal - minVal), -minVal/(maxVal - minVal));
    disparity_vis.convertTo(disparity_vis, CV_8UC1, 255.0);
    // do not use "disparity_vis" for depth calculation because disp values are relative
    cv::imwrite(disparity_out_path, disparity_vis);



    // --- convert to depth map
    std::string depth_map_out_path = "/workspace/results/" + dataset + "_depth_map.png";
    cv::Mat depth_map = computeDepthMapAndVis(disparity_gt, focal_length, baseline, depth_map_out_path);

    std::string mesh_output_path = "/workspace/results/" + dataset + "_mesh.off";
    generateMeshFromDepth(depth_map, cam0, dataset);

    return 0;
}
