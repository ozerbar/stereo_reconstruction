#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <iomanip> // For setw etc. alignment
#include <opencv2/calib3d.hpp>  
#include <opencv2/features2d.hpp> 
#include <opencv2/ximgproc.hpp>  
#include <opencv2/imgproc.hpp>
#include <fstream>


/*
* @brief Holds data of a COLMAP camera.
* @note Taken from ETH3D privded format loader https://github.com/ETH3D/format-loader/tree/master
*/
struct ColmapCamera {
  // Unique camera id.
  int camera_id;
  
  // Name of the distortion model. Determines the number of parameters.
  std::string model_name;
  
  // Image width in pixels.
  int width;
  
  // Image height in pixels.
  int height;
  
  // Distortion parameters. Their number and interpretation depends on the
  // distortion model.
  double parameters[4];
};


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
    double scaleFactor = std::sqrt(2) / avgDist;

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
 * @return             Returns the sum of squared errors (example)
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



/**
 * @brief Estimate the fundamental matrix using RANSAC
 * @param pointsLeft   All matched points in the left image
 * @param pointsRight  All matched points in the right image
 * @param iterations   Number of RANSAC iterations
 * @param threshold    Epipolar constraint threshold (e.g., 1~3, depends on actual scenario)
 * @return             The best F found by RANSAC
 */
cv::Mat ransacFundamentalMatrix(
    const std::vector<cv::Point2f> &pointsLeft,
    const std::vector<cv::Point2f> &pointsRight,
    int iterations = 1000,
    double threshold = 0.5)
{
    // Print the current number of input points
    std::cout << "[RANSAC] pointsLeft.size() = " << pointsLeft.size() 
              << ", pointsRight.size() = " << pointsRight.size() << std::endl;

    if (pointsLeft.size() != pointsRight.size() || pointsLeft.size() < 8)
    {
        std::cerr << "[ransacFundamentalMatrix] Not enough matching points or mismatch size!" << std::endl;
        return cv::Mat();
    }

    int maxInliers = -1;
    cv::Mat bestF;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Randomly select 8 distinct indices from [0, n-1]
    std::uniform_int_distribution<> dis(0, static_cast<int>(pointsLeft.size() - 1));

    for (int iter = 0; iter < iterations; ++iter)
    {
        // (1) Randomly pick 8 distinct points
        std::vector<cv::Point2f> sampleLeft, sampleRight;
        sampleLeft.reserve(8);
        sampleRight.reserve(8);

        std::set<int> usedIndices; // Prevent duplicates
        while ((int)usedIndices.size() < 8)
        {
            int idx = dis(gen);
            usedIndices.insert(idx);
        }
        for (auto idx : usedIndices)
        {
            sampleLeft.push_back(pointsLeft[idx]);
            sampleRight.push_back(pointsRight[idx]);
        }

        // (2) Use these 8 pairs of points to estimate F
        cv::Mat F_candidate = computeFundamentalMatrix(sampleLeft, sampleRight);
        if (F_candidate.empty()) 
            continue;

        // (3) Compute the error for all points and count inliers
        int inlierCount = 0;
        for (size_t i = 0; i < pointsLeft.size(); ++i)
        {
            cv::Mat x1 = (cv::Mat_<double>(3, 1) << pointsLeft[i].x, pointsLeft[i].y, 1.0);
            cv::Mat x2 = (cv::Mat_<double>(3, 1) << pointsRight[i].x, pointsRight[i].y, 1.0);

            double error = std::abs(x2.dot(F_candidate * x1));
            if (error < threshold)
                inlierCount++;
        }

        // (4) Update the best F if this has more inliers
        if (inlierCount > maxInliers)
        {
            maxInliers = inlierCount;
            bestF = F_candidate.clone();
        }
    }

    std::cout << "[RANSAC] Max inliers found: " << maxInliers << " / " 
              << pointsLeft.size() << std::endl;
    return bestF;
}


/*
* @brief Parse the camera intrinsics from the cameras.txt file
* @param cameraIndex   The index of the camera to parse (0-3)
* @param cameraTxtPath The path to the cameras.txt file
* @return              The parsed camera intrinsics
* @note                The camera idex for current images left.jpg and right.jpg is 1
*/

ColmapCamera parseCameraIntrinsicsLeftRight(const int cameraIndex, const std::string& cameraTxtPath)
{
    printf("Test\n");
    if (cameraIndex < 0 || cameraIndex > 3) {
        std::cerr << "Error: Invalid camera index." << std::endl;
        exit(-1);
    }

    // Load cameras.txt file
    std::ifstream txtFile(cameraTxtPath);
    if (!txtFile.is_open()) {
        std::cerr << "Error: Could not open camera intrinsics file." << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(txtFile, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            std::cerr << "Debug: Skipping comment or empty line." << std::endl;
            continue;
        }

        std::istringstream lineStream(line);
        int currentCameraId;
        if (!(lineStream >> currentCameraId)) {
            std::cerr << "Debug: Failed to parse camera ID in line: " << line << std::endl;
            continue;
        }

        std::cerr << "Debug: Line " << line << std::endl;

        if (currentCameraId == cameraIndex) {
            ColmapCamera camera;
            camera.camera_id = currentCameraId;
            
            // Read all values explicitly
            if (!(lineStream >> camera.model_name 
                          >> camera.width 
                          >> camera.height 
                          >> camera.parameters[0]  // fx
                          >> camera.parameters[1]  // fy
                          >> camera.parameters[2]  // cx
                          >> camera.parameters[3])) {  // cy
                std::cerr << "Error: Failed to parse camera parameters." << std::endl;
                exit(-1);
            }
            std::cerr << "Debug: Camera parameters parsed successfully." << std::endl;
            return camera;
        }
    }

    std::cerr << "Error: Camera index " << cameraIndex << " not found." << std::endl;
    exit(-1);
}



/**
 * @brief Compute disparity map from two rectified images using Block Matching
 * @param leftImage     The left rectified image
 * @param rightImage    The right rectified image
 * @return              The computed disparity map
 */
cv::Mat computeDisparityMap(const cv::Mat& leftImage, const cv::Mat& rightImage)
{
    // Parameters for block matching (BM)
    int numDisparities = 16;  // Maximum disparity
    int blockSize = 9;        // Block size for BM

    // Create StereoBM object for block matching
    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);

    // Compute disparity map
    cv::Mat disparity;
    stereoBM->compute(leftImage, rightImage, disparity);

    return disparity;
}



/**
 * @brief Convert disparity map to depth map using camera parameters
 * @param disparity     The disparity map
 * @param focalLength   The focal length (fx) of the camera
 * @param baseline      The baseline (distance between two cameras)
 * @return              The computed depth map
 */
cv::Mat computeDepthMap(const cv::Mat& disparity, double focalLength, double baseline)
{
    // Initialize the depth map
    cv::Mat depthMap(disparity.size(), CV_64F);

    // Loop through each pixel to calculate depth
    for (int y = 0; y < disparity.rows; ++y)
    {
        for (int x = 0; x < disparity.cols; ++x)
        {
            double d = disparity.at<float>(y, x);  // Disparity value at pixel (x, y)

            // Avoid division by zero or very small disparity values
            if (d > 0)
            {
                // Depth = (focalLength * baseline) / disparity
                double depth = (focalLength * baseline) / d;
                depthMap.at<double>(y, x) = depth;
            }
            else
            {
                // Assign zero depth to invalid disparity values
                depthMap.at<double>(y, x) = 0.0;
            }
        }
    }

    return depthMap;
}


// ----------------- DECOMPOSE ESSENTIAL MATRIX ----------------- //

/**
 * @brief Check if a 3D point is in front of a camera
 * @param P         The camera projection matrix
 * @param point3D   The 3D point in homogeneous coordinates
 * @return          True if the point is in front of the camera
 */
bool isInFrontOfCamera(const cv::Mat& P, const cv::Mat& point3D) {
    // Convert point to homogeneous coordinates if not already
    cv::Mat homogeneous_point;
    if (point3D.rows == 3)
        homogeneous_point = cv::Mat::ones(4, 1, CV_64F);
    else
        homogeneous_point = point3D.clone();
    
    if (point3D.rows == 3) {
        point3D.copyTo(homogeneous_point.rowRange(0, 3));
    }

    // Project point
    cv::Mat projected = P * homogeneous_point;
    
    // Get depth (Z coordinate)
    double depth = projected.at<double>(2) / projected.at<double>(3);
    
    return depth > 0;
}


/**
 * @brief Triangulate a 3D point from two 2D points and two projection matrices
 * @param P1    The projection matrix of the first camera
 * @param P2    The projection matrix of the second camera
 * @param pt1   The first 2D point
 * @param pt2   The second 2D point
 * @return      The triangulated 3D point in homogeneous coordinates
 */
cv::Mat triangulatePoint(const cv::Mat& P1, const cv::Mat& P2, 
                        const cv::Point2f& pt1, const cv::Point2f& pt2) {
    cv::Mat A(4, 4, CV_64F);
    
    // For first point
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    
    // For second point
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);
    
    cv::SVD svd(A);
    cv::Mat point3D = svd.vt.row(3).t();
    
    // Convert to homogeneous coordinates
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
 * @note This function tries all possible combinations of R and T to find the best solution
 *       based on the number of points in front of both cameras
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

    // Possible rotations and translations
    std::vector<cv::Mat> R_possibilities{
        svd.u * W * svd.vt,        // R1
        svd.u * W.t() * svd.vt,    // R2
    };
    std::vector<cv::Mat> t_possibilities{
        svd.u.col(2),              // t1
        -svd.u.col(2)              // t2
    };

    // Fix rotation matrices if needed (determinant = 1)
    for (auto& R : R_possibilities) {
        if (cv::determinant(R) < 0) {
            R = -R;
        }
    }

    // Create projection matrix for first camera
    cv::Mat P1 = cv::Mat::eye(4, 4, CV_64F);
    P1 = P1(cv::Rect(0, 0, 4, 3));  // Take first 3 rows to make 3x4

    int max_points_in_front = 0;
    int best_R_idx = 0;
    int best_t_idx = 0;

    // Test all combinations
    for (size_t r = 0; r < R_possibilities.size(); ++r) {
        for (size_t t = 0; t < t_possibilities.size(); ++t) {
            // Create second camera projection matrix
            cv::Mat P2 = cv::Mat::eye(4, 4, CV_64F);
            R_possibilities[r].copyTo(P2(cv::Rect(0, 0, 3, 3)));
            t_possibilities[t].copyTo(P2(cv::Rect(3, 0, 1, 3)));
            P2 = P2(cv::Rect(0, 0, 4, 3));  // Take first 3 rows to make 3x4

            int points_in_front = 0;
            
            // Check for each points if it is in front of both cameras, and update the best solution for R and T if needed
            for (size_t i = 0; i < points1.size(); ++i) {
                cv::Mat point3D = triangulatePoint(P1, P2, points1[i], points2[i]);
                
                // Check if point is in front of both cameras
                if (isInFrontOfCamera(P1, point3D) && 
                    isInFrontOfCamera(P2, point3D)) {
                    points_in_front++;
                }
            }

            // Update best solution 
            if (points_in_front > max_points_in_front) {
                max_points_in_front = points_in_front;
                best_R_idx = r;
                best_t_idx = t;
            }
        }
    }

    // Set the best R and t
    best_R = R_possibilities[best_R_idx].clone();
    best_t = t_possibilities[best_t_idx].clone();

    std::cout << "Selected solution has " << max_points_in_front << " points in front of both cameras" 
              << " out of " << points1.size() << " total points." << std::endl;
}

// ----------------- END DECOMPOSE ESSENTIAL MATRIX ----------------- //



/**
 * @brief Perform stereo rectification and compute disparity map from stereo images
 * @param F           The fundamental matrix between the two cameras
 * @param K1          The intrinsic matrix of the left camera
 * @param K2          The intrinsic matrix of the right camera
 * @param leftImage   The left input image
 * @param rightImage  The right input image
 * @note              This function first computes the essential matrix from the fundamental matrix 
 *                    and intrinsic matrices, then uses SVD to extract the rotation and translation
 *                    matrices for stereo rectification. After rectifying the images, it computes 
 *                    the disparity map using the Block Matching (BM) algorithm.
 */
void stereoRectifyAndComputeDisparity(const cv::Mat& F, const cv::Mat& K1, const cv::Mat& K2,
                                      const cv::Mat& leftImage, const cv::Mat& rightImage,
                                      const std::vector<cv::Point2f>& pointsLeft,
                                      const std::vector<cv::Point2f>& pointsRight)
{
    // 1. Compute the essential matrix E from the fundamental matrix F and camera matrices K1, K2
    cv::Mat E = K2.t() * F * K1;
    std::cout << "Essential matrix E:\n" << E << std::endl;

    // 2. Use SVD to decompose E into rotation and translation (R and T)
    // cv::SVD svd(E);
    // cv::Mat U = svd.u, S = svd.w, Vt = svd.vt;

    // cv::Mat R = U * cv::Mat::eye(3, 3, CV_64F) * Vt;
    // cv::Mat T = U.col(2);
    
    // NEW: Decompose essential matrix into R and T with trying out all combinations of R and T
    cv::Mat T, R;
    decomposeEssentialMatrix(E, pointsLeft, pointsRight, R, T);

    std::cout << "Rotation matrix R:\n" << R << std::endl;
    std::cout << "Translation vector T:\n" << T << std::endl;


    // 3. Perform stereo rectification (R1, R2, P1, P2, Q)
    // DEBUG: Add error checking for matrix types and dimensions
    CV_Assert(!K1.empty() && !K2.empty() && K1.size() == cv::Size(3,3) && K2.size() == cv::Size(3,3));
    CV_Assert(!leftImage.empty() && !rightImage.empty());

    // DEBUG
    std::cout << "Debug Image size: " << leftImage.size() << std::endl;
    std::cout << "Debug K1: " << K1.type() << " size: " << K1.size() << std::endl;
    std::cout << "Debug K2: " << K2.type() << " size: " << K2.size() << std::endl;

    cv::Mat R1, R2, P1, P2, Q;  // rectification/rotation matrices (R1, R2), projection matrices (P1, P2), disparity-to-depth mapping matrix (Q)
    
    try {
        cv::stereoRectify(K1, cv::Mat(), K2, cv::Mat(), leftImage.size(), R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1.0);

        // DEBUG
        std::cout << "Debug R1: " << R1.size() << " type: " << R1.type() << std::endl;
        std::cout << "Debug P1: " << P1.size() << " type: " << P1.type() << std::endl;


        // 4. Rectify the images using initUndistortRectifyMap and remap
        cv::Mat map1x, map1y, map2x, map2y;

        // cv::Mat K1_float, K2_float;
        // K1.convertTo(K1_float, CV_32F);
        // K2.convertTo(K2_float, CV_32F);

        cv::initUndistortRectifyMap(K1, cv::Mat(), R1, P1, leftImage.size(), CV_32FC1, map1x, map1y); 
        cv::initUndistortRectifyMap(K2, cv::Mat(), R2, P2, rightImage.size(), CV_32FC1, map2x, map2y);

        cv::Mat rectifiedLeft, rectifiedRight;
        cv::remap(leftImage, rectifiedLeft, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(rightImage, rectifiedRight, map2x, map2y, cv::INTER_LINEAR);

        // 5. Compute the disparity map using block matching
        cv::Mat disparity = computeDisparityMap(rectifiedLeft, rectifiedRight);
        
        // 6. Display the rectified images and disparity map
        //如果没有安装qt插件,就不能使用imshow在线预览,因此改为了直接写入文件,可以在本地查看
        // cv::imshow("Rectified Left", rectifiedLeft);
        // cv::imshow("Rectified Right", rectifiedRight);
        // cv::imshow("Disparity", disparity);
        // cv::waitKey(0);

        cv::imwrite("results/rectified_left.jpg", rectifiedLeft);
        cv::imwrite("results/rectified_right.jpg", rectifiedRight);
        cv::imwrite("results/disparity_map.jpg", disparity);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}



int main()
{
   // 1. Load images
    std::string leftImagePath  = "/workspace/Datasets/courtyard_dslr_undistorted/courtyard/images/left.jpg";
    std::string rightImagePath = "/workspace/Datasets/courtyard_dslr_undistorted/courtyard/images/right.jpg";

    cv::Mat leftImage  = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
    if (leftImage.empty() || rightImage.empty())
    {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // 2. Use ORB to detect and describe features
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;
    orb->detectAndCompute(leftImage,  cv::noArray(), keypointsLeft,  descriptorsLeft);
    orb->detectAndCompute(rightImage, cv::noArray(), keypointsRight, descriptorsRight);

    // 3. Use BFMatcher for feature matching (note crossCheck=true reduces number of matches)
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // If you want more matches, set crossCheck=false and do more sophisticated filtering

    // 4. Sort matches by distance
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &m1, const cv::DMatch &m2)
              {
                  return m1.distance < m2.distance;
              });

    // 5. Extract matched point coordinates
    std::vector<cv::Point2f> ptsLeft, ptsRight;
    ptsLeft.reserve(matches.size());
    ptsRight.reserve(matches.size());
    for (auto &m : matches)
    {
        ptsLeft.push_back(keypointsLeft[m.queryIdx].pt);
        ptsRight.push_back(keypointsRight[m.trainIdx].pt);
    }

    // Ensure we have at least 8 matches for the fundamental matrix
    if (ptsLeft.size() < 8)
    {
        std::cerr << "Error: Not enough matches to compute Fundamental Matrix!" << std::endl;
        return -1;
    }

    // 6. Estimate F with all matching points (without RANSAC)
    cv::Mat F_all = computeFundamentalMatrix(ptsLeft, ptsRight);
    std::cout << "[Direct] F estimated from all points: \n" << F_all << std::endl;
    double error_all = computeReprojectionError(ptsLeft, ptsRight, F_all);
    std::cout << "[Direct] Reprojection error (sum of squares): " << error_all << std::endl;

    // 7. Estimate F via RANSAC
    cv::Mat F_ransac = ransacFundamentalMatrix(ptsLeft, ptsRight, /*iterations*/2000, /*threshold*/1.0);
    std::cout << "[RANSAC] F estimated: \n" << F_ransac << std::endl;
    double error_ransac = computeReprojectionError(ptsLeft, ptsRight, F_ransac);
    std::cout << "[RANSAC] Reprojection error (sum of squares): " << error_ransac << std::endl;

    // 8. Visualize some matched points
    const int numMatchesToDraw = 10; // e.g., draw 10 matches
    std::vector<cv::DMatch> topMatches(matches.begin(),
                                       matches.begin() + std::min<int>(matches.size(), numMatchesToDraw));

    cv::Mat matchImage;
    cv::drawMatches(leftImage, keypointsLeft,
                    rightImage, keypointsRight,
                    topMatches, matchImage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Draw thicker lines as a demonstration
    for (auto &m : topMatches)
    {
        cv::Point2f ptLeft  = keypointsLeft[m.queryIdx].pt;
        cv::Point2f ptRight = keypointsRight[m.trainIdx].pt;
        // Since we draw on a single image, shift the right image's coordinates by leftImage.cols
        ptRight.x += (float)leftImage.cols;

        cv::line(matchImage, ptLeft, ptRight, cv::Scalar(0, 255, 0), 2);
    }

    // 9. Save the visualization
    std::string outputPath = "output_with_opencv.jpg";
    if (!cv::imwrite(outputPath, matchImage))
    {
        std::cerr << "Error: Could not save the match image." << std::endl;
        return -1;
    }
    std::cout << "Match image saved to " << outputPath << std::endl;
  

    // 10. Load camera intrinsics from cameras.txt
    std::string cameraTxtPath = "/workspace/Datasets/courtyard_dslr_undistorted/courtyard/dslr_calibration_undistorted/cameras.txt";
    // std::cerr << "Debug: File path is " << cameraTxtPath << std::endl;获取得到的路径
    ColmapCamera camera = parseCameraIntrinsicsLeftRight(1, cameraTxtPath);
    std::cout << "Camera intrinsics for camera index 1:\n"
              << "Model: " << camera.model_name << "\n"
              << "Width: " << camera.width << "\n"
              << "Height: " << camera.height << "\n"
              << "Parameters: fx=" << camera.parameters[0] << ", fy=" << camera.parameters[1]
              << ", cx=" << camera.parameters[2] << ", cy=" << camera.parameters[3] << std::endl;
    
    // Convert camera intrinsics to intrinsics matrix K
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        camera.parameters[0], 0, camera.parameters[2],
        0, camera.parameters[1], camera.parameters[3],
        0, 0, 1);


    // 11. Perform stereo rectification and compute disparity 
    stereoRectifyAndComputeDisparity(F_all, K, K, leftImage, rightImage, ptsLeft, ptsRight); 

    return 0;
}


