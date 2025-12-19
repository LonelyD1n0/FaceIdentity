#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

void readImageAsGrayFloat(const std::string &path, cv::Mat &out, int width = 100, int height = 100);
void loadDataset(const std::string &datasetDir, std::vector<cv::Mat> &images, std::vector<int> &labels, std::vector<std::string> &labelNames, int width = 100, int height = 100);
bool isImageFile(const std::string &path);
// Preprocess arbitrary image (color or gray) to single-channel CV_64F of size width x height.
void preprocessImage(const cv::Mat &src, cv::Mat &out, int width = 100, int height = 100);
