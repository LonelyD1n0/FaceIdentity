#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd imagesToEigen(const std::vector<cv::Mat> &imgs);
void computePCA(const MatrixXd &X, int K, VectorXd &meanVec, MatrixXd &eigenVectors, MatrixXd &projections);
cv::Mat projectImage(const cv::Mat &img, const VectorXd &meanVec, const MatrixXd &eigenVectors);
cv::Mat projectionsToCvMat(const MatrixXd &proj);
