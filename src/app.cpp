#include "app.h"
#include "dataset.h"
#include "pca.h"
#include <filesystem>
#include <iostream>
#include <opencv2/imgproc.hpp>
// include face module if available (Facemark requires opencv_contrib)
#if defined(__has_include)
#  if __has_include(<opencv2/face.hpp>)
#    include <opencv2/face.hpp>
#    define HAVE_OPENCV_FACE 1
#  endif
#endif

using namespace std;
using namespace cv;
using namespace cv::ml;
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::string locateCascade(const char* argv0, const std::string &cascadeName) {
    std::vector<std::string> candidates = {
        cascadeName,
        (std::string("../") + cascadeName),
        (std::string("data/haarcascades/") + cascadeName),
        (std::string("data/") + cascadeName),
        (std::string("haarcascades/") + cascadeName),
        (std::string("../share/opencv4/haarcascades/") + cascadeName),
        (std::string("../share/opencv/haarcascades/") + cascadeName),
        std::string("C:/Users/60152/vcpkg/installed/x64-windows/share/opencv4/haarcascades/") + cascadeName
    };

    for (auto &p : candidates) if (std::filesystem::exists(p)) return p;

    std::filesystem::path cwd = std::filesystem::current_path();
    for (int depth = 0; depth < 4; ++depth) {
        for (auto &p : candidates) {
            auto full = (cwd / p).lexically_normal();
            if (std::filesystem::exists(full)) return full.string();
        }
        if (!cwd.has_parent_path()) break;
        cwd = cwd.parent_path();
    }

    if (argv0) {
        std::filesystem::path exepath = std::filesystem::absolute(argv0).parent_path();
        std::filesystem::path cur = exepath;
        for (int depth = 0; depth < 4; ++depth) {
            for (auto &p : candidates) {
                auto full = (cur / p).lexically_normal();
                if (std::filesystem::exists(full)) return full.string();
            }
            if (!cur.has_parent_path()) break;
            cur = cur.parent_path();
        }
    }
    return string();
}

int doTrain(const string &datasetDir, const string &outModel, int comps, int imgW, int imgH) {
    cout << "Loading dataset from: " << datasetDir << endl;
    vector<Mat> images; vector<int> labels; vector<string> labelNames;
    loadDataset(datasetDir, images, labels, labelNames, imgW, imgH);
    if (images.empty()) { cerr << "No images found." << endl; return -1; }

    MatrixXd X = imagesToEigen(images);
    VectorXd meanVec; MatrixXd eigenVecs, projections;
    cout << "Computing PCA (components=" << comps << ")..." << endl;
    computePCA(X, comps, meanVec, eigenVecs, projections);
    cout << "PCA done. feature dim=" << eigenVecs.cols() << endl;

    Mat trainData = projectionsToCvMat(projections);
    Mat trainLabels((int)labels.size(), 1, CV_32S);
    for (size_t i = 0; i < labels.size(); ++i) trainLabels.at<int>((int)i,0) = labels[i];

    cout << "Training SVM..." << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    Mat trainData32;
    if (trainData.type() != CV_32F) trainData.convertTo(trainData32, CV_32F);
    else trainData32 = trainData;
    svm->train(trainData32, ROW_SAMPLE, trainLabels);
    svm->save(outModel);
    cout << "SVM saved to: " << outModel << endl;

    FileStorage fs((outModel + ".pca.yml"), FileStorage::WRITE);
    Mat meanMat(meanVec.rows(), 1, CV_64F);
    for (int i = 0; i < meanVec.rows(); ++i) meanMat.at<double>(i,0) = meanVec(i);
    Mat eigMat(eigenVecs.rows(), eigenVecs.cols(), CV_64F);
    for (int c = 0; c < eigenVecs.cols(); ++c) for (int r = 0; r < eigenVecs.rows(); ++r) eigMat.at<double>(r,c) = eigenVecs(r,c);
    fs << "mean" << meanMat;
    fs << "eigs" << eigMat;
    fs << "labels" << "[";
    for (auto &n : labelNames) fs << n;
    fs << "]";
    fs.release();
    cout << "PCA data saved to: " << outModel << ".pca.yml" << endl;
    return 0;
}

int doPredict(const string &predictPath, Ptr<SVM> svm, const VectorXd &meanVec, const MatrixXd &eigenVecs, CascadeClassifier &face_cascade, const vector<string> &labelNames, int imgW, int imgH) {
    cout << "Predicting image: " << predictPath << endl;
    if (!std::filesystem::exists(predictPath)) { cerr << "File not found: " << predictPath << endl; return -1; }

    Mat img = imread(predictPath, IMREAD_COLOR);
    if (img.empty()) { cerr << "Failed to read image: " << predictPath << endl; return -1; }

    Mat gray; cvtColor(img, gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    if (!face_cascade.empty()) face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30,30));
    Mat faceROI = faces.empty() ? gray.clone() : gray(faces[0]).clone();
    Mat work; resize(faceROI, work, Size(imgW, imgH));
    work.convertTo(work, CV_64F, 1.0/255.0);
    Mat feat = projectImage(work, meanVec, eigenVecs);
    int pred = (int)svm->predict(feat);
    float decision = svm->predict(feat, noArray(), StatModel::RAW_OUTPUT);
    string predName = (pred >= 0 && pred < (int)labelNames.size()) ? labelNames[pred] : to_string(pred);
    cout << "Prediction: " << predName << " (label=" << pred << ")" << " decision=" << decision << endl;
    return 0;
}

int doTest(const string &testDir, Ptr<SVM> svm, const VectorXd &meanVec, const MatrixXd &eigenVecs, CascadeClassifier &face_cascade, const vector<string> &labelNames, int imgW, int imgH) {
    cout << "Running test on: " << testDir << endl;
    vector<Mat> testImages; vector<int> testLabels; vector<string> testLabelNames;
    loadDataset(testDir, testImages, testLabels, testLabelNames, imgW, imgH);
    if (testImages.empty()) { cerr << "No test images found in: " << testDir << endl; return -1; }

    int N = (int)testImages.size();
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        Mat feat = projectImage(testImages[i], meanVec, eigenVecs);
        int pred = (int)svm->predict(feat);
        float decision = svm->predict(feat, noArray(), StatModel::RAW_OUTPUT);
        string trueName = testLabelNames[testLabels[i]];
        auto it = find(labelNames.begin(), labelNames.end(), trueName);
        int trueIdx = (it == labelNames.end()) ? -1 : (int)distance(labelNames.begin(), it);
        string predName = (pred >= 0 && pred < (int)labelNames.size()) ? labelNames[pred] : to_string(pred);
        cout << "Image " << i << ": true=" << trueName << " pred=" << predName << (pred==trueIdx ? " OK" : " FAIL") << " decision=" << decision << endl;
        if (pred == trueIdx) ++correct;
    }
    double acc = 100.0 * correct / N;
    cout << "Test done. Accuracy: " << acc << "% (" << correct << "/" << N << ")" << endl;
    return 0;
}

int runCamera(Ptr<SVM> svm, const VectorXd &meanVec, const MatrixXd &eigenVecs, CascadeClassifier &face_cascade, const vector<string> &labelNames, int imgW, int imgH) {
    cout << "Starting camera. Press 'q' to quit." << endl;
    VideoCapture cap(0);
    if (!cap.isOpened()) { cerr << "Failed to open camera." << endl; return -1; }

    // try to locate and load facemark model (LBF) for robust landmark detection (optional)
#if defined(HAVE_OPENCV_FACE)
    cv::Ptr<cv::face::Facemark> facemark;
    string lbfModel = locateCascade(nullptr, "lbfmodel.yaml");
    if (!lbfModel.empty()) {
        try {
            facemark = cv::face::FacemarkLBF::create();
            facemark->loadModel(lbfModel);
            cout << "Loaded facemark model: " << lbfModel << endl;
        } catch (const cv::Exception &e) {
            cerr << "Failed to load facemark model: " << e.what() << endl;
            facemark.release();
        }
    }
#else
    // facemark (opencv_contrib face) not available in this build
    cv::Ptr<void> facemark; (void)facemark;
#endif

    // fallback: try to locate and load an eye cascade for lower-quality heuristic
    CascadeClassifier eye_cascade;
    string eyePath = locateCascade(nullptr, "haarcascade_eye.xml");
    if (eyePath.empty()) eyePath = locateCascade(nullptr, "haarcascade_eye_tree_eyeglasses.xml");
    if (!eyePath.empty()) {
        if (eye_cascade.load(eyePath)) cout << "Loaded eye cascade: " << eyePath << endl;
        else cerr << "Warning: failed to load eye cascade: " << eyePath << endl;
    } else {
        cerr << "Warning: eye cascade not found; low-head detection may be degraded." << endl;
    }

    // simple GUI mode selector: 0 = identity, 1 = low-head
    const string winName = "Face Recognition";
    int mode = 0; // default identity
    namedWindow(winName, WINDOW_AUTOSIZE);
    createTrackbar("Mode (0:ID,1:LowHead)", winName, &mode, 1);

    Mat frame;
    while (true) {
        cap >> frame; if (frame.empty()) break;
        Mat gray; cvtColor(frame, gray, COLOR_BGR2GRAY);
        Mat gray_eq; equalizeHist(gray, gray_eq);

        vector<Rect> faces;
        if (!face_cascade.empty()) {
            face_cascade.detectMultiScale(gray_eq, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(40,40));
            if (faces.empty()) face_cascade.detectMultiScale(gray_eq, faces, 1.05, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));
            if (faces.empty()) face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30,30));
        } else {
            faces.push_back(Rect(0,0,gray.cols, gray.rows));
        }

        vector<Rect> toProcess;
        if (!faces.empty()) {
            auto it = std::max_element(faces.begin(), faces.end(), [](const Rect&a,const Rect&b){ return a.area()<b.area(); });
            toProcess.push_back(*it);
        } else {
            int w = gray.cols, h = gray.rows;
            int cw = w/2, ch = h/2;
            int x = max(0, (w - cw)/2);
            int y = max(0, (h - ch)/2);
            Rect centerRect(x, y, min(cw, w - x), min(ch, h - y));
            toProcess.push_back(centerRect);
        }

        for (auto &r : toProcess) {
            Mat face = gray(r).clone();
            // detect depending on mode
            if (mode == 1) {
                bool lowHead = false;
                bool didLandmarks = false;
                // prefer facemark landmarks if available
#if defined(HAVE_OPENCV_FACE)
                if (facemark) {
                    std::vector<std::vector<cv::Point2f>> shapes;
                    std::vector<Rect> singleFace = { r };
                    try {
                        if (facemark->fit(frame, singleFace, shapes) && !shapes.empty()) {
                            auto &shape = shapes[0];
                            // compute eyes center and nose tip
                            // 68-point model indices: left eye 36-41, right eye 42-47, nose tip ~30
                            double eyeY = 0.0; int eyeCount = 0;
                            for (int i = 36; i <= 41; ++i) { eyeY += shape[i].y; ++eyeCount; }
                            for (int i = 42; i <= 47; ++i) { eyeY += shape[i].y; ++eyeCount; }
                            eyeY /= (double)eyeCount;
                            double noseY = shape[30].y;
                            double norm = (double)r.height;
                            double rel = (noseY - eyeY) / norm; // normalized distance
                            // if nose is significantly lower than eyes -> head down
                            if (rel > 0.10) lowHead = true;
                            didLandmarks = true;
                        }
                    } catch (const cv::Exception &e) {
                        // facemark may fail on some frames; continue to fallback
                        didLandmarks = false;
                    }
                }
#else
                (void)facemark; // no-op when facemark not available
#endif

                if (!didLandmarks) {
                    // fallback: use eye cascade heuristic
                    vector<Rect> eyes;
                    if (!eye_cascade.empty()) {
                        eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(10,10));
                    }
                    if (!eyes.empty()) {
                        double avgY = 0.0;
                        for (auto &e : eyes) avgY += (e.y + e.height*0.5);
                        avgY /= (eyes.size() * (double)face.rows);
                        if (avgY > 0.55) lowHead = true;
                    } else {
                        double ar = (double)face.rows / (double)face.cols;
                        if (ar > 1.3) lowHead = true;
                    }
                }

                // draw and annotate
                rectangle(frame, r, lowHead ? Scalar(0,0,255) : Scalar(0,255,0), 2);
                string label = lowHead ? "LOW HEAD" : "UP";
                putText(frame, label, Point(r.x, r.y - 6), FONT_HERSHEY_SIMPLEX, 0.9, lowHead ? Scalar(0,0,255) : Scalar(0,255,0), 2);
            } else {
                // identity mode (original behavior)
                Mat resizedFace;
                resize(face, resizedFace, Size(imgW, imgH));
                resizedFace.convertTo(resizedFace, CV_64F, 1.0/255.0);
                Mat feat = projectImage(resizedFace, meanVec, eigenVecs);
                int pred = (int)svm->predict(feat);
                float decision = svm->predict(feat, noArray(), StatModel::RAW_OUTPUT);
                string name = (pred >= 0 && pred < (int)labelNames.size()) ? labelNames[pred] : to_string(pred);
                rectangle(frame, r, Scalar(0,255,0), 2);
                std::ostringstream oss; oss << name << " (" << decision << ")";
                putText(frame, oss.str(), Point(r.x, r.y - 6), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
            }
        }

        imshow("Face Recognition", frame);
        char c = (char)waitKey(10);
        if (c == 'q' || c == 27) break;
    }

    cap.release(); destroyAllWindows();
    return 0;
}
