/**
 * @file FaceDetector.h
 * @brief Объявление класса FaceDetector.
 */

#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/opencv.hpp>
#include "Image.h"

extern const std::string FACE_DETECTOR_MODEL_PATH;

/**
 * @class FaceDetector
 * @brief Класс для обнаружения лиц на изображениях с использованием каскадного классификатора.
 * Также этот класс рисует рамку и текст предсказания на изображении.
 */
class FaceDetector {

public:
    /**
     * @brief Конструктор загружает каскадный классификатор для обнаружения лиц.
     */
    FaceDetector();

    /**
     * @brief Обнаружение лиц на изображении и рисование рамок.
     * @param frame Изображение, на котором нужно обнаружить лица.
     */
    void detectFace(cv::Mat& frame);

    /**
     * @brief Рисует рамку вокруг обнаруженных лиц на изображении.
     * @param frame Изображение, на котором нужно нарисовать рамку.
     * @return Изображение с нанесенными рамками и областью интереса (ROI).
     */
    Image drawBoundingBoxOnFrame(cv::Mat& frame);

    /**
     * @brief Печатает текст с предсказанием эмоций на изображении.
     * @param image_and_ROI Изображение с рамками вокруг лиц.
     * @param emotion_prediction Вектор строк с предсказанными эмоциями.
     * @return Изображение с текстом предсказаний.
     */
    Image printPredictionTextToFrame(Image& image_and_ROI, std::vector<std::string>& emotion_prediction);

private:
    cv::CascadeClassifier cascade; ///< Каскадный классификатор для обнаружения лиц.
    std::vector<cv::Rect> faces; ///< Результаты обнаружения лиц.
};

#endif