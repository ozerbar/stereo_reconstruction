#ifndef FAST_SGBM_H
#define FAST_SGBM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

using CostType = uint16_t;  // 16-bit is typical for SGM. Could also do uint32_t if you prefer.

class FastSGBM
{
public:
    FastSGBM(int rows, int cols, int d_range,
             int P1, int P2,
             bool applyBlur = false);

    void compute_disp(const cv::Mat &leftGray,
                      const cv::Mat &rightGray,
                      cv::Mat &dispOut);

private:
    void censusTransform(const cv::Mat &img, cv::Mat &census) const;
    inline unsigned char hammingDist8(unsigned char a, unsigned char b) const;

    // Step 1: build pixelwise cost volume from Census
    void buildCostVolume(const cv::Mat &censusL, const cv::Mat &censusR);

    // Step 2: do 4-pass SGM aggregation
    void aggregateCosts();

    // Core SGM function that walks a direction (dr, dc)
    void aggregateOneDirection(int dr, int dc);

    // Step 3: winner-take-all disparity
    void selectDisparity(cv::Mat &dispOut);

private:
    int rows_, cols_;
    int disp_range_;
    int P1_, P2_;
    bool blur_;

    // Storage for census images
    cv::Mat censusL_, censusR_;

    // The raw pixelwise cost: cost_[r][c][d]
    // stored as a 1D array of size (rows_ * cols_ * disp_range_)
    std::vector<CostType> cost_;

    // The aggregated cost after 4 directions: sumCost_[r][c][d]
    std::vector<CostType> sumCost_;
};

#endif
