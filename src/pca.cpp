#include "pca.h"
#include <opencv2/opencv.hpp>

using namespace cv;

// 转换成Eigen矩阵
MatrixXd imagesToEigen(const std::vector<cv::Mat> &imgs) {
    int N = (int)imgs.size();
    if (N == 0) return MatrixXd();
    int D = imgs[0].rows * imgs[0].cols;
    MatrixXd X(D, N);
    for (int i = 0; i < N; ++i) {
        cv::Mat tmp = imgs[i].reshape(1, D); // D x 1
        for (int r = 0; r < D; ++r) X(r, i) = tmp.at<double>(r, 0);
    }
    return X;
}
// 计算均值与主成分
void computePCA(const MatrixXd &X, int K, VectorXd &meanVec, MatrixXd &eigenVectors, MatrixXd &projections) {
    int D = (int)X.rows();
    int N = (int)X.cols();
    meanVec = X.rowwise().mean(); // D x 1
    MatrixXd Xc = X.colwise() - meanVec; // D x N

    MatrixXd Csmall = Xc.transpose() * Xc; // N x N

    Eigen::SelfAdjointEigenSolver<MatrixXd> es(Csmall);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed");

    VectorXd evals = es.eigenvalues();
    MatrixXd evecs_small = es.eigenvectors();

    K = std::min(K, N);
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return evals[a] > evals[b]; });

    eigenVectors = MatrixXd(D, K);
    projections = MatrixXd(K, N);

    for (int k = 0; k < K; ++k) {
        int i = idx[k];
        double lambda = evals(i);
        if (lambda <= 1e-12) {
            eigenVectors.col(k).setZero();
            continue;
        }
        VectorXd v = evecs_small.col(i);
        VectorXd u = Xc * v;
        u /= sqrt(lambda);
        u.normalize();
        eigenVectors.col(k) = u;
    }

    projections = eigenVectors.transpose() * Xc;
}

cv::Mat projectImage(const cv::Mat &img, const VectorXd &meanVec, const MatrixXd &eigenVectors) {
    int D = img.rows * img.cols;
    cv::Mat tmp = img.reshape(1, D);
    MatrixXd x(D,1);
    for (int r = 0; r < D; ++r) x(r,0) = tmp.at<double>(r,0);
    MatrixXd x_center = x.col(0) - meanVec;
    MatrixXd feat = eigenVectors.transpose() * x_center; // K x 1
    cv::Mat out(1, feat.rows(), CV_32F);
    for (int i = 0; i < feat.rows(); ++i) out.at<float>(0,i) = float(feat(i,0));
    return out;
}

cv::Mat projectionsToCvMat(const MatrixXd &proj) {
    int K = (int)proj.rows();
    int N = (int)proj.cols();
    cv::Mat out(N, K, CV_32F);
    for (int i = 0; i < N; ++i) for (int j = 0; j < K; ++j) out.at<float>(i,j) = float(proj(j,i));
    return out;
}
