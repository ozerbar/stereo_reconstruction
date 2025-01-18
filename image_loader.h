#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>


/**
 * @brief Holds data of a COLMAP camera (example struct).
 */
struct ColmapCamera {
    int camera_id;
    std::string model_name;
    int width;
    int height;
    double parameters[4]; // e.g., fx, fy, cx, cy
};

/**
 * @brief Load an image (e.g., "im0.png") in GRAYSCALE.
 * @param imagePath The path to the image file
 * @return          Grayscale image as cv::Mat
 */
inline cv::Mat loadImageGrayscale(const std::string& imagePath)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
    }
    return img;
}

/**
 * @brief Parse a camera's intrinsics from a "camera.txt" style file.
 * @note  If the file has lines like:
 *         cam1=[fx cx fy cy]
 *        or something similar, adapt as needed.
 *
 * @param cameraIndex   The index of the camera to parse (e.g., 0,1,...)
 * @param cameraTxtPath The path to the file
 * @return              The ColmapCamera struct with intrinsics set
 */
inline ColmapCamera parseCameraIntrinsicsLeftRight(const int cameraIndex,
                                                   const std::string& cameraTxtPath)
{
    std::ifstream txtFile(cameraTxtPath);
    if (!txtFile.is_open()) {
        std::cerr << "Error: Could not open camera intrinsics file: "
                  << cameraTxtPath << std::endl;
        std::exit(-1);
    }

    ColmapCamera camera;
    camera.camera_id = cameraIndex;
    std::string line;

    // Key for the desired camera, e.g. "cam1="
    std::string camKey = "cam" + std::to_string(cameraIndex) + "=";
    bool foundCamera = false;

    while (std::getline(txtFile, line)) {
        if (line.find(camKey) != std::string::npos) {
            // Found the line for this camera
            foundCamera = true;

            // Example: cam1=[7293.188 0 123.4; 0 7293.188 234.5; 0 0 1]
            size_t start = line.find('[');
            size_t end   = line.find(']');
            if (start == std::string::npos || end == std::string::npos) 
                break;

            std::string matrixData = line.substr(start + 1, end - (start + 1));
            std::istringstream iss(matrixData);

            double fx, fy, cx, cy;
            // Typically you might see: fx >> cx >> fy >> cy
            // or some arrangement. Adjust if your format differs.
            iss >> fx >> cx >> fy >> cy;

            camera.model_name = "Custom";
            camera.parameters[0] = fx;
            camera.parameters[1] = fy;
            camera.parameters[2] = cx;
            camera.parameters[3] = cy;
        }
        else if (line.find("width=") != std::string::npos) {
            camera.width = std::stoi(line.substr(line.find('=') + 1));
        }
        else if (line.find("height=") != std::string::npos) {
            camera.height = std::stoi(line.substr(line.find('=') + 1));
        }
    }

    if (!foundCamera) {
        std::cerr << "Error: Camera index " << cameraIndex 
                  << " not found in file " << cameraTxtPath << std::endl;
        std::exit(-1);
    }

    return camera;
}

/**
 * @brief Read calibration parameters from a file with lines like:
 *
 *   cam0=[fx 0 cx; 0 fy cy; 0 0 1]
 *   cam1=[fx 0 cx; 0 fy cy; 0 0 1]
 *   doffs=...
 *   baseline=...
 *   width=...
 *   height=...
 *   ndisp=...
 *   isint=...
 *   vmin=...
 *   vmax=...
 *   dyavg=...
 *   dymax=...
 *
 * @param filename      Path to calib.txt
 * @param cameraMatrix0 (OUT) 3x3 double
 * @param cameraMatrix1 (OUT) 3x3 double
 * @param doffs         (OUT)
 * @param baseline      (OUT)
 * @param width         (OUT)
 * @param height        (OUT)
 * @param ndisp         (OUT)
 * @param isint         (OUT)
 * @param vmin          (OUT)
 * @param vmax          (OUT)
 * @param dyavg         (OUT)
 * @param dymax         (OUT)
 */
inline void loadCalibFile(
    const std::string &filename,
    cv::Mat &cameraMatrix0,
    cv::Mat &cameraMatrix1,
    double &doffs,
    double &baseline,
    int &width,
    int &height,
    int &ndisp,
    int &isint,
    int &vmin,
    int &vmax,
    double &dyavg,
    double &dymax
) {
    // Ensure 3x3 double
    cameraMatrix0 = cv::Mat::zeros(3,3, CV_64F);
    cameraMatrix1 = cv::Mat::zeros(3,3, CV_64F);

    std::ifstream fin(filename);
    if(!fin.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    std::string line;
    while(std::getline(fin, line)) {
        if(line.empty()) continue;
        
        // cam0=
        if(line.find("cam0=") != std::string::npos) {
            // parse matrix values
            auto pos = line.find('=');
            std::string values = line.substr(pos+1);
            values.erase(std::remove(values.begin(), values.end(), '['), values.end());
            values.erase(std::remove(values.begin(), values.end(), ']'), values.end());
            values.erase(std::remove(values.begin(), values.end(), ';'), values.end());

            std::istringstream iss(values);
            double arr[9];
            int idx = 0;
            while(iss >> arr[idx]) { 
                idx++; 
                if(idx>=9) break; 
            }
            // fill cameraMatrix0
            int r=0, c=0;
            for(int i=0; i<9; i++) {
                cameraMatrix0.at<double>(r,c) = arr[i];
                c++; 
                if(c==3) { c=0; r++; }
            }
        }
        // cam1=
        else if(line.find("cam1=") != std::string::npos) {
            auto pos = line.find('=');
            std::string values = line.substr(pos+1);
            values.erase(std::remove(values.begin(), values.end(), '['), values.end());
            values.erase(std::remove(values.begin(), values.end(), ']'), values.end());
            values.erase(std::remove(values.begin(), values.end(), ';'), values.end());

            std::istringstream iss(values);
            double arr[9];
            int idx = 0;
            while(iss >> arr[idx]) {
                idx++;
                if(idx>=9) break;
            }
            int r=0, c=0;
            for(int i=0; i<9; i++) {
                cameraMatrix1.at<double>(r,c) = arr[i];
                c++;
                if(c==3) { c=0; r++; }
            }
        }
        else if(line.find("doffs=") != std::string::npos) {
            doffs = std::stod(line.substr(line.find('=')+1));
        }
        else if(line.find("baseline=") != std::string::npos) {
            baseline = std::stod(line.substr(line.find('=')+1));
        }
        else if(line.find("width=") != std::string::npos) {
            width = std::stoi(line.substr(line.find('=')+1));
        }
        else if(line.find("height=") != std::string::npos) {
            height = std::stoi(line.substr(line.find('=')+1));
        }
        else if(line.find("ndisp=") != std::string::npos) {
            ndisp = std::stoi(line.substr(line.find('=')+1));
        }
        else if(line.find("isint=") != std::string::npos) {
            isint = std::stoi(line.substr(line.find('=')+1));
        }
        else if(line.find("vmin=") != std::string::npos) {
            vmin = std::stoi(line.substr(line.find('=')+1));
        }
        else if(line.find("vmax=") != std::string::npos) {
            vmax = std::stoi(line.substr(line.find('=')+1));
        }
        else if(line.find("dyavg=") != std::string::npos) {
            dyavg = std::stod(line.substr(line.find('=')+1));
        }
        else if(line.find("dymax=") != std::string::npos) {
            dymax = std::stod(line.substr(line.find('=')+1));
        }
    }
    fin.close();
}

