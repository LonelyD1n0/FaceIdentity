/*
Face recognition (PCA + SVM)
Single-file C++ example using OpenCV + Eigen

Dependencies:
 - OpenCV (>=3.4 recommended)
 - Eigen3
 - C++17

Build (example with CMake):
 1) Create CMakeLists.txt that links OpenCV and sets include for Eigen3
 2) mkdir build && cd build
 3) cmake .. && make

Data layout (training):
 dataset/
   person1/
     img1.jpg
     img2.jpg
   person2/
     img1.jpg
   ...

Usage:
 - Run with --train <dataset_dir> --model model.xml --components 50
 - After training, run without --train to start camera recognition: --model model.xml

This file contains:
 - loadImages: read images and labels from dataset folders
 - PCA implementation (Eigen)
 - SVM training and prediction (OpenCV ml::SVM)
 - camera recognition demo

Note: This is an educational example; for production use, handle exceptions and edge cases.
*/

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "pca.h"
#include "svm.h"
#include "app.h"
#include <algorithm>

namespace fs = std::filesystem;

using namespace std;
using namespace cv;
using namespace cv::ml;

// --- Parameters ---
const int IMG_W = 100;
const int IMG_H = 100;

int main(int argc, char** argv) {
    string dataset = "face"; // default dataset folder at project root
    string modelPath = "face_svm.xml";
    bool doTraining = false;
    int components = 50;
    string testDir = "";
    string predictPath = "";
    string cascadeArg = "";

    // simple arg parse
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--train" && i+1 < argc) { dataset = argv[++i]; doTraining = true; }
        else if (a == "--model" && i+1 < argc) modelPath = argv[++i];
        else if (a == "--components" && i+1 < argc) components = stoi(argv[++i]);
    else if (a == "--test" && i+1 < argc) testDir = argv[++i];
    else if (a == "--predict" && i+1 < argc) predictPath = argv[++i];
    else if (a == "--cascade" && i+1 < argc) cascadeArg = argv[++i];
        else if (a == "--help") { cout << "Usage: --train <dataset_dir> --model <model.xml> --components <K> --test <test_dir>\n"; return 0; }
    }

    // If user did not explicitly request training but model is missing and default dataset exists,
    // automatically train using the default 'face' folder.
    if (!doTraining) {
        if (!fs::exists(modelPath) && fs::exists(dataset)) {
            cout << "Model '" << modelPath << "' not found and dataset '" << dataset << "' exists.\n";
            cout << "Starting training automatically using dataset: " << dataset << "\n";
            doTraining = true;
        }
    }

    if (doTraining) {
        return doTrain(dataset, modelPath, components, IMG_W, IMG_H);
    }

    // Recognition mode
    // load PCA & SVM
    if (!fs::exists(modelPath)) { cerr << "Model not found: " << modelPath << "\nUse --train to train first." << endl; return -1; }
    cout << "Loading SVM: " << modelPath << endl;
    Ptr<ml::SVM> svm = ml::SVM::load(modelPath);
    if (!svm) { cerr << "Failed to load SVM model: " << modelPath << endl; return -1; }

    // load PCA YAML
    FileStorage pfs((modelPath + ".pca.yml"), FileStorage::READ);
    Mat meanMat, eigMat; vector<string> labelNames;
    pfs["mean"] >> meanMat;
    pfs["eigs"] >> eigMat;
    FileNode lnode = pfs["labels"];
    for (auto it = lnode.begin(); it != lnode.end(); ++it) labelNames.push_back((string)*it);
    pfs.release();

    // convert mean and eigs to Eigen
    VectorXd meanVec(meanMat.rows);
    for (int i = 0; i < meanMat.rows; ++i) meanVec(i) = meanMat.at<double>(i,0);
    MatrixXd eigenVecs(eigMat.rows, eigMat.cols);
    for (int c = 0; c < eigMat.cols; ++c) for (int r = 0; r < eigMat.rows; ++r) eigenVecs(r,c) = eigMat.at<double>(r,c);

    // load cascade (helper in app.cpp)
    CascadeClassifier face_cascade;
    string cascade_path;
    if (!cascadeArg.empty()) {
        if (std::filesystem::exists(cascadeArg)) cascade_path = cascadeArg;
        else cerr << "Warning: cascade file provided but not found: " << cascadeArg << endl;
    }
    if (cascade_path.empty()) cascade_path = locateCascade(argv[0], "haarcascade_frontalface_default.xml");
    if (!cascade_path.empty()) {
        if (face_cascade.load(cascade_path)) cout << "Loaded face cascade: " << cascade_path << endl;
        else cerr << "Warning: cascade file found but failed to load: " << cascade_path << endl;
    } else {
        cerr << "Warning: failed to locate 'haarcascade_frontalface_default.xml'. Face detection may fail." << endl;
    }

    // try to locate eye cascade for low-head detection and provide to app (app will locate again too)
    CascadeClassifier eye_cascade;
    string eye_path = locateCascade(argv[0], "haarcascade_eye.xml");
    if (eye_path.empty()) eye_path = locateCascade(argv[0], "haarcascade_eye_tree_eyeglasses.xml");
    if (!eye_path.empty()) {
        if (eye_cascade.load(eye_path)) cout << "Loaded eye cascade: " << eye_path << endl;
        else cerr << "Warning: eye cascade found but failed to load: " << eye_path << endl;
    } else {
        cerr << "Note: eye cascade not found in search paths; app will attempt a secondary search." << endl;
    }

    // dispatch to helpers (now implemented in src/app.cpp)
    if (!predictPath.empty()) return doPredict(predictPath, svm, meanVec, eigenVecs, face_cascade, labelNames, IMG_W, IMG_H);
    if (!testDir.empty()) return doTest(testDir, svm, meanVec, eigenVecs, face_cascade, labelNames, IMG_W, IMG_H);
    return runCamera(svm, meanVec, eigenVecs, face_cascade, labelNames, IMG_W, IMG_H);
}
