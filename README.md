# Rectification and Disparity computation based on opencv 

    - all implements are in stereo-reconstruction.cpp

    - run the cpp file you will get 6 images for Rectification(3 method of keypoint matching **orb, sift, brisk** with respect to 2 Rectification method **calibrated and uncalibrated**) also 6 images for Disparity map(3 keypoint matching methods with respect to 2 Rectification methods)

    - pipiline: 
        -1.use three keypoint matching methods and respectively compute it Fundamental matrix with ransac.
        -2. for calibrated method, we need the intrisics and extrinsics. So first get the camera intrisics and do decomposition on ransan Fundamental matrix to get rotation and translation
        -3. for uncalibrated method, we dont need the intrisics and extrinsics

    - if you want to change the dataset, first **change the image load path** and then **the camera intrisic matrix**

