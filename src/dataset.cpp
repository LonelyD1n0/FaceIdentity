#include "dataset.h"
#include <filesystem>
#include <iostream>
#include <algorithm>

static void letterboxOrCrop(const cv::Mat &src, cv::Mat &dst, int targetW, int targetH) {
    // Preserve aspect ratio. If aspect differs, we will resize then center-crop or pad to fit target.
    int srcW = src.cols, srcH = src.rows;
    double scale = std::max((double)targetW / srcW, (double)targetH / srcH);
    // Use scale to ensure the resized image covers the target, then center-crop
    int resizeW = (int)std::round(srcW * scale);
    int resizeH = (int)std::round(srcH * scale);
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(resizeW, resizeH));
    // center crop
    int x = std::max(0, (resizeW - targetW) / 2);
    int y = std::max(0, (resizeH - targetH) / 2);
    cv::Rect roi(x, y, targetW, targetH);
    if (x + targetW <= tmp.cols && y + targetH <= tmp.rows) dst = tmp(roi).clone();
    else {
        // fallback: pad
        dst = cv::Mat::zeros(targetH, targetW, tmp.type());
        int copyW = std::min(targetW, tmp.cols);
        int copyH = std::min(targetH, tmp.rows);
        tmp(cv::Rect(0,0,copyW,copyH)).copyTo(dst(cv::Rect(0,0,copyW,copyH)));
    }
}

void preprocessImage(const cv::Mat &src, cv::Mat &out, int width, int height) {
    cv::Mat gray;
    if (src.channels() == 1) gray = src;
    else if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else {
        // convert to 8-bit gray via first channel as fallback
        std::vector<cv::Mat> chs;
        cv::split(src, chs);
        gray = chs[0];
    }

    // resize with aspect-preserve and center crop
    cv::Mat resized;
    letterboxOrCrop(gray, resized, width, height);

    // convert to double and scale to [0,1]
    resized.convertTo(out, CV_64F, 1.0/255.0);
}

static std::string toLower(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
    return out;
}

bool isImageFile(const std::string &path) {
    std::string ext = toLower(std::filesystem::path(path).extension().string());
    static const std::vector<std::string> exts = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

namespace fs = std::filesystem;

void readImageAsGrayFloat(const std::string &path, cv::Mat &out, int width, int height) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("Failed to read: " + path);
    preprocessImage(img, out, width, height);
}

void loadDataset(const std::string &datasetDir, std::vector<cv::Mat> &images, std::vector<int> &labels, std::vector<std::string> &labelNames, int width, int height) {
    images.clear(); labels.clear(); labelNames.clear();
    int label = 0;
    for (auto &p : fs::directory_iterator(datasetDir)) {
        if (!p.is_directory()) continue;
        std::string person = p.path().filename().string();
        labelNames.push_back(person);
        for (auto &f : fs::directory_iterator(p.path())) {
            if (!f.is_regular_file()) continue;
            std::string fp = f.path().string();
            if (!isImageFile(fp)) continue; // skip non-image files
            try {
                cv::Mat img; readImageAsGrayFloat(fp, img, width, height);
                images.push_back(img);
                labels.push_back(label);
            } catch (const std::exception &e) {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
        }
        label++;
    }
}
