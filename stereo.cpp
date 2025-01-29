#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

#include <vector>
#include <iostream>

void normalizePoints(const std::vector<cv::Point2f> &points,
                     std::vector<cv::Point2f> &normalizedPoints,
                     cv::Mat &T)
{
    // Compute centroid
    cv::Point2f center(0, 0);
    for (const auto &p : points)
    {
        center += p;
    }
    center *= (1.0 / points.size());

    // Compute average distance
    double avgDist = 0;
    for (const auto &p : points)
    {
        avgDist += cv::norm(p - center);
    }
    avgDist /= points.size();

    // Scale factor
    double scaleFactor = std::sqrt(2) / avgDist;

    // Construct the transformation matrix
    T = (cv::Mat_<double>(3, 3) << scaleFactor, 0, -scaleFactor * center.x,
         0, scaleFactor, -scaleFactor * center.y,
         0, 0, 1);

    // Normalize the points
    normalizedPoints.resize(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::Mat p = (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, 1.0);
        cv::Mat p_norm = T * p;
        normalizedPoints[i].x = p_norm.at<double>(0) / p_norm.at<double>(2);
        normalizedPoints[i].y = p_norm.at<double>(1) / p_norm.at<double>(2);
    }
}

cv::Mat computeFundamentalMatrix(const std::vector<cv::Point2f> &pointsLeft,
                                 const std::vector<cv::Point2f> &pointsRight)
{
    size_t N = pointsLeft.size();
    if (N < 8)
    {
        std::cerr << "[computeFundamentalMatrix] Not enough points!" << std::endl;
        return cv::Mat();
    }

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

    // SVD for computing fundamental matrix
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::VectorXd f = V.col(8);
    cv::Mat F(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i)
    {
        F.at<double>(i / 3, i % 3) = f(i);
    }

    // Enforce rank-2
    cv::SVD svdF(F);
    cv::Mat U = svdF.u;
    cv::Mat W = svdF.w;
    cv::Mat Vt = svdF.vt;
    W.at<double>(2, 0) = 0.0;
    F = U * cv::Mat::diag(W) * Vt;

    // Denormalize the fundamental matrix
    F = T2.t() * F * T1;

    return F;
}
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
    double error = std::abs(A * p2.x + B * p2.y + C) / std::sqrt(A * A + B * B);

    return error;
}
cv::Mat ransacFundamentalMatrix(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, int maxIterations, float threshold, int &inlierCount)
{
    int bestInlierCount = 0;
    cv::Mat bestF;
    srand(time(0));

    // RANSAC process
    for (int i = 0; i < maxIterations; ++i)
    {
        std::vector<cv::Point2f> samplePoints1, samplePoints2;
        for (int j = 0; j < 8; ++j)
        {
            int idx = rand() % points1.size();
            samplePoints1.push_back(points1[idx]);
            samplePoints2.push_back(points2[idx]);
        }

        cv::Mat F = computeFundamentalMatrix(samplePoints1, samplePoints2);

        std::vector<cv::Point2f> inliers1, inliers2;
        int count = 0;
        for (size_t j = 0; j < points1.size(); ++j)
        {
            double error = computeEpipolarError(points1[j], points2[j], F);
            if (error < threshold)
            {
                inliers1.push_back(points1[j]);
                inliers2.push_back(points2[j]);
                count++;
            }
        }

        if (count > bestInlierCount)
        {
            bestInlierCount = count;
            bestF = F.clone();
        }
    }

    inlierCount = bestInlierCount;
    return bestF;
}

int main()
{
    // Load images
    cv::Mat leftImage = cv::imread("/workspace/Datasets/left1.jpg", cv::IMREAD_COLOR);  // query image
    cv::Mat rightImage = cv::imread("/workspace/Datasets/left2.jpg", cv::IMREAD_COLOR); // train image
    if (leftImage.empty() || rightImage.empty())
    {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // Initialize variables for keypoints and descriptors
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    // Initialize ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    // Detect keypoints
    orb->detect(leftImage, keypointsLeft);
    orb->detect(rightImage, keypointsRight);

    // Compute descriptors
    orb->compute(leftImage, keypointsLeft, descriptorsLeft);
    orb->compute(rightImage, keypointsRight, descriptorsRight);

    // Match descriptors using Hamming Distance
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Select max and min distance, and define the good match
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    // Filter good matches based on distance
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // Collect matched points
    std::vector<cv::Point2f> ptsLeft, ptsRight;
    for (unsigned int i = 0; i < good_matches.size(); ++i)
    {
        ptsLeft.push_back(keypointsLeft[good_matches[i].queryIdx].pt);
        ptsRight.push_back(keypointsRight[good_matches[i].trainIdx].pt);
    }

    // Compute Fundamental Matrix using RANSAC
    int inlierCount = 0;
    cv::Mat F = ransacFundamentalMatrix(ptsLeft, ptsRight, 3000, 1.3, inlierCount);

    std::cout << "[RANSAC] F estimated: " << F << std::endl;

    // Compute rectification matrices using stereoRectifyUncalibrated
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(ptsLeft, ptsRight, F, leftImage.size(), H1, H2);

    // Apply rectification transformations
    cv::Mat rectifiedLeft, rectifiedRight;
    cv::warpPerspective(leftImage, rectifiedLeft, H1, leftImage.size());
    cv::warpPerspective(rightImage, rectifiedRight, H2, rightImage.size());

    // Save the rectified images
    cv::imwrite("rectified_left.jpg", rectifiedLeft);
    cv::imwrite("rectified_right.jpg", rectifiedRight);

    // Compute disparity map using StereoBM (block matching)
    cv::Mat leftGray, rightGray;
    cv::cvtColor(rectifiedLeft, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectifiedRight, rightGray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9); // 16 disparity levels and block size 9
    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity); // Compute disparity

    // Normalize disparity to view it as an image
    cv::Mat disparityNormalized;
    double minVal, maxVal;
    cv::minMaxLoc(disparity, &minVal, &maxVal);
    disparity.convertTo(disparityNormalized, CV_8U, 255.0 / (maxVal - minVal));

    // Save the disparity image
    cv::imwrite("disparity_image.jpg", disparityNormalized);

    std::cout << "Rectified left and right images and disparity map saved successfully." << std::endl;

    return 0;
}
