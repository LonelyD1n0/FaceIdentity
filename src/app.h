#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <Eigen/Dense>

std::string locateCascade(const char* argv0, const std::string &cascadeName);
int doTrain(const std::string &datasetDir, const std::string &outModel, int comps, int imgW, int imgH);
int doPredict(const std::string &predictPath, cv::Ptr<cv::ml::SVM> svm, const Eigen::VectorXd &meanVec, const Eigen::MatrixXd &eigenVecs, cv::CascadeClassifier &face_cascade, const std::vector<std::string> &labelNames, int imgW, int imgH);
int doTest(const std::string &testDir, cv::Ptr<cv::ml::SVM> svm, const Eigen::VectorXd &meanVec, const Eigen::MatrixXd &eigenVecs, cv::CascadeClassifier &face_cascade, const std::vector<std::string> &labelNames, int imgW, int imgH);
int runCamera(cv::Ptr<cv::ml::SVM> svm, const Eigen::VectorXd &meanVec, const Eigen::MatrixXd &eigenVecs, cv::CascadeClassifier &face_cascade, const std::vector<std::string> &labelNames, int imgW, int imgH);
