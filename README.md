# Rectification and Disparity computation based on opencv 

    - All implements are in stereo-reconstruction.cpp

    - run the cpp file you will get 6 images for Rectification(3 method of keypoint matching **orb, sift, brisk** with respect to 2 Rectification method **calibrated and uncalibrated**) also 6 images for Disparity map(3 keypoint matching methods with respect to 2 Rectification methods)

    - Pipiline: 
        1. Use three sparse keypoint matching methods and respectively compute the Fundamental matrix with ransac.
        2. For calibrated rectification method, we need the intrisics and extrinsics. So first get the camera intrisics and do decomposition on ransan Fundamental matrix to get rotation and translation.
        3. For uncalibrated rectification method, we dont need the intrisics and extrinsics.
        4. We calculate the disparity base dof the rectified images.
        5. We calculate the depth map based of the disparity map.
            - multiple versions are available (with/without opencv)
        6. We generate the point clouds and the 3D mesh from the depth map.

    - If you want to change the dataset, first **change the image load path** and then **the camera intrisic matrix**. You would also have to adjust the parameters:
        - Baseline and focal length of camera (contained in dataset like camer instrinsics)
        - Distance thresholds and valid edge thresholds for mesh generation (set to your own liking)
    

