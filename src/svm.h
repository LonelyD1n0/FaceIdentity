#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

void trainSVMandSave(const cv::Mat &trainData, const cv::Mat &trainLabels, const std::string &modelPath);
cv::Ptr<cv::ml::SVM> loadSVM(const std::string &modelPath);
