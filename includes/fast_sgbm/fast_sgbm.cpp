#include "fast_sgbm.h"
#include <algorithm>
#include <iostream>
#include <limits>

// Convenient indexer macro
#define COST_IDX(r,c,d) ((r)*(cols_)*(disp_range_) + (c)*(disp_range_) + (d))

FastSGBM::FastSGBM(int rows, int cols, int d_range, int P1, int P2, bool applyBlur)
    : rows_(rows), cols_(cols), disp_range_(d_range),
      P1_(P1), P2_(P2), blur_(applyBlur)
{
    // Allocate cost volumes
    cost_.resize(rows_ * cols_ * disp_range_, 0);
    sumCost_.resize(rows_ * cols_ * disp_range_, 0);

    censusL_ = cv::Mat(rows_, cols_, CV_8UC1);
    censusR_ = cv::Mat(rows_, cols_, CV_8UC1);
}

void FastSGBM::compute_disp(const cv::Mat &leftGray,
                            const cv::Mat &rightGray,
                            cv::Mat &dispOut)
{
    // 1) Possibly blur
    cv::Mat left = leftGray, right = rightGray;
    if (blur_) {
        cv::GaussianBlur(leftGray, left, cv::Size(3,3), 0);
        cv::GaussianBlur(rightGray, right, cv::Size(3,3), 0);
    }

    // 2) Census transform
    censusTransform(left, censusL_);
    censusTransform(right, censusR_);

    // 3) Build pixelwise cost volume (Hamming distance of 8-bit Census)
    std::fill(cost_.begin(), cost_.end(), 0);
    buildCostVolume(censusL_, censusR_);

    // 4) Aggregate along 4 directions
    std::fill(sumCost_.begin(), sumCost_.end(), 0);
    // top-left -> bottom-right
    aggregateOneDirection(+1, +1);
    // top-right -> bottom-left
    aggregateOneDirection(+1, -1);
    // bottom-left -> top-right
    aggregateOneDirection(-1, +1);
    // bottom-right -> top-left
    aggregateOneDirection(-1, -1);

    // 5) Winner-take-all
    dispOut.create(rows_, cols_, CV_8UC1);
    selectDisparity(dispOut);
}

// Census over a 3x3 neighborhood
void FastSGBM::censusTransform(const cv::Mat &img, cv::Mat &census) const
{
    census.setTo(0);
    for (int r = 1; r < rows_ - 1; r++) {
        for (int c = 1; c < cols_ - 1; c++) {
            unsigned char center = img.at<uchar>(r, c);
            unsigned char val = 0;
            val |= (img.at<uchar>(r-1, c-1) >= center) << 7;
            val |= (img.at<uchar>(r-1, c  ) >= center) << 6;
            val |= (img.at<uchar>(r-1, c+1) >= center) << 5;
            val |= (img.at<uchar>(r,   c+1) >= center) << 4;
            val |= (img.at<uchar>(r+1, c+1) >= center) << 3;
            val |= (img.at<uchar>(r+1, c  ) >= center) << 2;
            val |= (img.at<uchar>(r+1, c-1) >= center) << 1;
            val |= (img.at<uchar>(r,   c-1) >= center) << 0;

            census.at<uchar>(r, c) = val;
        }
    }
}

inline unsigned char FastSGBM::hammingDist8(unsigned char a, unsigned char b) const
{
    unsigned char x = a ^ b;
    // builtin popcount for 8 bits if you want:
    // return (unsigned char)__builtin_popcount(x);
    unsigned char dist = 0;
    while (x) {
        x &= (x - 1);
        dist++;
    }
    return dist;
}

// Build the raw cost volume: cost(r,c,d) = HammingDist of Census
void FastSGBM::buildCostVolume(const cv::Mat &censusL, const cv::Mat &censusR)
{
    for (int r = 0; r < rows_; r++) {
        const uchar* rowL = censusL.ptr<uchar>(r);
        for (int c = 0; c < cols_; c++) {
            unsigned char cL = rowL[c];
            for (int d = 0; d < disp_range_; d++) {
                int cr = c - d;
                unsigned char cR = 0;
                if (cr >= 0) {
                    cR = censusR.at<uchar>(r, cr);
                }
                cost_[COST_IDX(r,c,d)] = hammingDist8(cL, cR);
            }
        }
    }
}

// Helper: compute aggregated costs in one direction (dr, dc) using standard SGM min/second-min trick
void FastSGBM::aggregateOneDirection(int dr, int dc)
{
    // Temporary 2D array to store the aggregated cost for the *current* pass
    // aggregator(r,c,d) is same size as cost_.
    std::vector<CostType> aggregator(rows_ * cols_ * disp_range_, 0);

    // We choose iteration order depending on (dr, dc)
    // For example, if (dr, dc) = (1,1), we start from top-left and go down to bottom-right
    int rowStart = (dr > 0) ? 0 : (rows_ - 1);
    int rowEnd   = (dr > 0) ? rows_ : -1;
    int colStart = (dc > 0) ? 0 : (cols_ - 1);
    int colEnd   = (dc > 0) ? cols_ : -1;

    for (int r = rowStart; r != rowEnd; r += dr) {
        for (int c = colStart; c != colEnd; c += dc) {
            // cost at (r,c,d)
            // if there's no predecessor pixel, aggregator = cost
            CostType *outLine = &aggregator[COST_IDX(r,c,0)];
            const CostType *pixLine = &cost_[COST_IDX(r,c,0)];

            // Coordinates of previous pixel along direction (dr, dc)
            int pr = r - dr;
            int pc = c - dc;

            if (pr < 0 || pr >= rows_ || pc < 0 || pc >= cols_) {
                // no previous => aggregator(r,c,d) = cost(r,c,d)
                for (int d = 0; d < disp_range_; d++) {
                    outLine[d] = pixLine[d];
                }
            } else {
                // We do the standard formula:
                // aggregator(r,c,d) = cost(r,c,d) + min(
                //    aggregator(pr,pc,d),
                //    aggregator(pr,pc,d±1) + P1,
                //    min_of_previous + P2
                // ) - min_of_previous
                // 
                // BUT we use the min/second-min approach to get "cost_other" in O(1) instead of O(d_range).

                // 1) find minCost, secondMinCost, bestDisparity in aggregator(pr, pc, :)
                const CostType *prevLine = &aggregator[COST_IDX(pr, pc, 0)];

                CostType minCost = std::numeric_limits<CostType>::max();
                CostType secondMinCost = std::numeric_limits<CostType>::max();
                int bestD = -1;

                for (int d = 0; d < disp_range_; d++) {
                    CostType val = prevLine[d];
                    if (val < minCost) {
                        secondMinCost = minCost;
                        minCost = val;
                        bestD = d;
                    } else if (val < secondMinCost) {
                        secondMinCost = val;
                    }
                }

                // 2) For each d, compute aggregator using the known formula in O(1)
                for (int d = 0; d < disp_range_; d++) {
                    CostType baseCost = prevLine[d];
                    // costSame = baseCost
                    // costPlus = aggregator(pr,pc,d±1)+P1
                    // costOther = (d == bestD) ? secondMinCost + P2 : (minCost + P2)

                    CostType val = baseCost;

                    // handle +/- 1
                    if (d > 0) {
                        CostType alt = (CostType)(prevLine[d-1] + P1_);
                        if (alt < val) val = alt;
                    }
                    if (d < disp_range_ - 1) {
                        CostType alt = (CostType)(prevLine[d+1] + P1_);
                        if (alt < val) val = alt;
                    }

                    // handle costOther
                    if (d == bestD) {
                        CostType alt = (CostType)(secondMinCost + P2_);
                        if (alt < val) val = alt;
                    } else {
                        CostType alt = (CostType)(minCost + P2_);
                        if (alt < val) val = alt;
                    }

                    // aggregator(r,c,d) = cost(r,c,d) + val - minCost
                    // (the standard SGM formula)
                    CostType res = (CostType)(pixLine[d] + val - minCost);
                    outLine[d] = res;
                }
            }
        }
    }

    // Finally add aggregator to sumCost_
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            for (int d = 0; d < disp_range_; d++) {
                sumCost_[COST_IDX(r,c,d)] += aggregator[COST_IDX(r,c,d)];
            }
        }
    }
}

// After summing costs from all directions, pick best d for each pixel
void FastSGBM::selectDisparity(cv::Mat &dispOut)
{
    for (int r = 0; r < rows_; r++) {
        uchar* rowPtr = dispOut.ptr<uchar>(r);
        for (int c = 0; c < cols_; c++) {
            CostType bestCost = sumCost_[COST_IDX(r,c,0)];
            int bestD = 0;
            for (int d = 1; d < disp_range_; d++) {
                CostType val = sumCost_[COST_IDX(r,c,d)];
                if (val < bestCost) {
                    bestCost = val;
                    bestD = d;
                }
            }
            rowPtr[c] = static_cast<uchar>(bestD);
        }
    }
}
