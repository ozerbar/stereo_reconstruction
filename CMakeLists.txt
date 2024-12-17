cmake_minimum_required(VERSION 3.6)

project(stereo_reconstruction)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(FLANN REQUIRED)

# Common headers
set(HEADERS
    Eigen.h
    stb_image.h
    stb_image_write.h
)

# 1) Executable: keypoint_matching_opencv
set(SRC_WITH_OPENCV
    keypoint_matching_with_opencv.cpp
    ${HEADERS}
)
add_executable(keypoint_matching_opencv ${SRC_WITH_OPENCV})
target_link_libraries(keypoint_matching_opencv
    Eigen3::Eigen
    ${OpenCV_LIBS}    # uses OpenCV for keypoint matching
    flann
    lz4
)

# 2) Executable: keypoint_matching_no_opencv
set(SRC_NO_OPENCV
    keypoint_matching_without_opencv.cpp
    ${HEADERS}
)
add_executable(keypoint_matching_no_opencv ${SRC_NO_OPENCV})
target_link_libraries(keypoint_matching_no_opencv
    Eigen3::Eigen
    flann
    lz4
)

# To build the 'main.cpp' for stereo_reconstruction:
# set(SOURCES main.cpp ${HEADERS})
# add_executable(stereo_reconstruction ${SOURCES})
# target_link_libraries(stereo_reconstruction
#     Eigen3::Eigen
#     ${OpenCV_LIBS}
#     flann
#     lz4
# )

