#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>

using namespace cv;
using namespace cv::ml;

void trainSVMandSave(const Mat &trainData, const Mat &trainLabels, const std::string &modelPath) {
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setC(2.0);
	svm->setGamma(0.05);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-6));
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	svm->save(modelPath);
}

Ptr<SVM> loadSVM(const std::string &modelPath) {
	return SVM::load(modelPath);
}
