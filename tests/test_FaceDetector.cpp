#pragma once

#define CATCH_CONFIG_MAIN
#include "../contrib/catch/catch.hpp"

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <mutex>
#include <filesystem>


#include "../src/Image.h"
#include "../src/FaceDetector.h"
#include "../src/Model.h"

const std::string FACE_DETECTOR_MODEL_PATH = "model/haarcascade_frontalface_alt2.xml";
const std::string TENSORFLOW_MODEL_PATH = "model/tensorflow_model.pb";
const std::string APP_NAME = "Real-Time Facial Emotion Recognition";
const std::string WAY = "";

TEST_CASE("Testing FaceDetector") {
    std::filesystem::path test_path = "src/image.jpg";
    std::filesystem::path error_test_path = "src/error_image.jpg";
    FaceDetector faceDetector;

    SECTION("Test detectFace with a simple image") {
        cv::Mat test_image = cv::imread(test_path);
        REQUIRE_FALSE(test_image.empty());

        faceDetector.detectFace(test_image);
        CHECK(faceDetector.faceCount() > 0);
    }

    SECTION("Test detectFace with a simple image") {
        cv::Mat test_image = cv::imread(error_test_path);
        REQUIRE_FALSE(test_image.empty());

        faceDetector.detectFace(test_image);
        CHECK(faceDetector.faceCount() == 0);
    }

    SECTION("Test drawBoundingBoxOnFrame") {
        cv::Mat test_image = cv::imread(test_path);
        REQUIRE_FALSE(test_image.empty());

        faceDetector.detectFace(test_image);
        Image result_image = faceDetector.drawBoundingBoxOnFrame(test_image);

        cv::Mat frame_with_boxes = result_image.getFrame();
        CHECK_FALSE(frame_with_boxes.empty());
    }

    SECTION("Test printPredictionTextToFrame") {
        cv::Mat test_image = cv::imread(test_path);
        REQUIRE_FALSE(test_image.empty());

        faceDetector.detectFace(test_image);
        Image result_image = faceDetector.drawBoundingBoxOnFrame(test_image);
        std::vector<std::string> predictions = {"Happy"};

        Image image_with_text = faceDetector.printPredictionTextToFrame(result_image, predictions);

        cv::Mat frame_with_text = image_with_text.getFrame();
        CHECK_FALSE(frame_with_text.empty());
    }
}

TEST_CASE("Testing FaceDetector full pipeline main emotion") {
    std::filesystem::path test_path = "src/image.jpg";

    Model model(TENSORFLOW_MODEL_PATH);
    FaceDetector face_detector;
    Image image_and_ROI;
    cv::Mat frame;

    frame = cv::imread(test_path);

    face_detector.detectFace(frame);
    image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

    std::vector<cv::Mat> roi_image = image_and_ROI.getROI();

    if (roi_image.size() > 0) {
        image_and_ROI.preprocessROI();

        std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
        std::string emotion_prediction_2 = model.ans(image_and_ROI);

        image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
        
        REQUIRE(emotion_prediction[0].find("Happy") != std::string::npos);
    } else {
        REQUIRE(true == false);
    }
}

TEST_CASE("Testing FaceDetector full pipeline second emotion") {
    std::filesystem::path test_path = "src/image.jpg";

    Model model(TENSORFLOW_MODEL_PATH);
    FaceDetector face_detector;
    Image image_and_ROI;
    cv::Mat frame;

    frame = cv::imread(test_path);

    face_detector.detectFace(frame);

    image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

    std::vector<cv::Mat> roi_image = image_and_ROI.getROI();

    if (roi_image.size() > 0) {
        image_and_ROI.preprocessROI();

        std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
        std::string emotion_prediction_2 = model.ans(image_and_ROI);

        image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
        
        REQUIRE(emotion_prediction_2.find("Neutral") != std::string::npos);
    } else {
        REQUIRE(true == false);
    }
}

TEST_CASE("FaceDetector loads model and detects faces") {
    std::filesystem::path test_path = "src/image.jpg";
    std::filesystem::path error_test_path = "src/error_image.jpg";
    FaceDetector faceDetector;

    SECTION("Correct image") {
        cv::Mat testImage = cv::imread(test_path);
        REQUIRE(!testImage.empty());

        faceDetector.detectFace(testImage);
        REQUIRE(faceDetector.faceCount() > 0);

        Image resultImage = faceDetector.drawBoundingBoxOnFrame(testImage);
        REQUIRE(!resultImage.getFrame().empty());

        std::vector<std::string> emotions = {"happy", "sad"};
        Image finalImage = faceDetector.printPredictionTextToFrame(resultImage, emotions);
        REQUIRE(!finalImage.getFrame().empty());
    }

    SECTION("Incorrect image") {
        cv::Mat testImage = cv::imread(error_test_path);
        REQUIRE(!testImage.empty());

        faceDetector.detectFace(testImage);
        REQUIRE(faceDetector.faceCount() == 0);
    }
}

TEST_CASE("Image class handles frames and ROIs",) {
    std::filesystem::path test_path = "src/image.jpg";
    std::filesystem::path error_test_path = "src/error_image.jpg";

    SECTION("Correct image") {
        Image image;

        cv::Mat testFrame = cv::imread(test_path);
        REQUIRE(!testFrame.empty());

        image.setFrame(testFrame);
        REQUIRE(!image.getFrame().empty());
        REQUIRE(image.getFrame().rows == testFrame.rows);
        REQUIRE(image.getFrame().cols == testFrame.cols);

        cv::Rect roiRect(0, 0, 100, 100);
        cv::Mat roi = testFrame(roiRect);
        image.setROI(roi);
        std::vector<cv::Mat> rois = image.getROI();
        REQUIRE(rois.size() > 0);
        REQUIRE(rois[0].rows == roi.rows);
        REQUIRE(rois[0].cols == roi.cols);

        image.preprocessROI();
        std::vector<cv::Mat> modelInput = image.getModelInput();
        REQUIRE(modelInput.size() > 0);
        REQUIRE(modelInput[0].channels() == 1);
    }

    SECTION("Incorrect image") {
        Image image;

        cv::Mat testFrame = cv::imread(error_test_path);
        REQUIRE(!testFrame.empty());

        image.setFrame(testFrame);
        REQUIRE(!image.getFrame().empty());
        REQUIRE(image.getFrame().rows == testFrame.rows);
        REQUIRE(image.getFrame().cols == testFrame.cols);

        cv::Rect roiRect(0, 0, 100, 100);
        cv::Mat roi = testFrame(roiRect);
        image.setROI(roi);
        std::vector<cv::Mat> rois = image.getROI();
        REQUIRE(rois.size() > 0);
        REQUIRE(rois[0].rows == roi.rows);
        REQUIRE(rois[0].cols == roi.cols);

        image.preprocessROI();
        std::vector<cv::Mat> modelInput = image.getModelInput();
        REQUIRE(modelInput.size() > 0);
        REQUIRE(modelInput[0].channels() == 1);
    }
}

TEST_CASE("Model loads and makes predictions") {
    std::filesystem::path test_path = "src/image.jpg";
    std::filesystem::path error_test_path = "src/error_image.jpg";

    Model model(TENSORFLOW_MODEL_PATH);

    SECTION("Correct image") {
        cv::Mat testFrame = cv::imread(test_path);
        REQUIRE(!testFrame.empty());
        Image image;
        image.setFrame(testFrame);

        std::vector<std::string> predictions = model.predict(image);
        REQUIRE(predictions.size() > 0);

        std::string answer = model.ans(image);
        REQUIRE(!answer.empty());
    }

    SECTION("Incorrect image") {
        cv::Mat testFrame = cv::imread(error_test_path);

        REQUIRE(!testFrame.empty());
        Image image;
        image.setFrame(testFrame);

        std::vector<std::string> predictions = model.predict(image);
        REQUIRE(predictions.size() == 0);

        std::string answer = model.ans(image);
        REQUIRE(answer.empty());

    }
}