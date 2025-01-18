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

#include "image_loader.h"
#include "stereo_rectify.h"

std::string DatasetName = "Shopvac-imperfect";


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


int main() {
    // 1) Load images
    auto leftImage  = loadImageGrayscale("../Datasets/"+DatasetName+"/im0.png");
    auto rightImage = loadImageGrayscale("../Datasets/"+DatasetName+"/im1.png");

    // 2) Load calibration
    cv::Mat cam0, cam1;
    double doffs, baseline;
    int width, height, ndisp, isint, vmin, vmax;
    double dyavg, dymax;
    loadCalibFile(
        "../Datasets/"+DatasetName+"/calib.txt",
        cam0, cam1,
        doffs,
        baseline,
        width,
        height,
        ndisp,
        isint,
        vmin,
        vmax,
        dyavg,
        dymax
    );

    // 3) Rectify
    cv::Mat rectLeft, rectRight;
    cv::Mat Q = rectifyStereoImages(
        leftImage,
        rightImage,
        rectLeft,
        rectRight,
        cam0,
        cam1,
        baseline,
        width,
        height
    );

    // 4) Show or save results
    showRectifiedPair(rectLeft, rectRight, width, height);

    // 5) e.g. compute disparity
    cv::Mat disparity = computeDisparityMap(rectLeft, rectRight);
    cv::imwrite("disparity.jpg", disparity);

    return 0;
}