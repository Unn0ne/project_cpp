/**
 * @file Image.cpp
 * @brief Реализация методов класса Image.
 */

#include <opencv2/opencv.hpp>
#include "Image.h"

/**
 * @brief Получает текущее изображение (кадр).
 * @return Текущее изображение.
 */
cv::Mat Image::getFrame() {
    return this->_frame;
}

/**
 * @brief Устанавливает текущее изображение (кадр).
 * @param frame Изображение, которое нужно установить.
 */
void Image::setFrame(cv::Mat& frame) {
    this->_frame = frame;
}

/**
 * @brief Получает вектор областей интереса (ROI).
 * @return Вектор изображений областей интереса.
 */
std::vector<cv::Mat> Image::getROI() {
    return this->_roi_image;
}

/**
 * @brief Получает вектор изображений для входа модели.
 * @return Вектор изображений для входа модели.
 */
std::vector<cv::Mat> Image::getModelInput() {
    return this->_model_input_image;
}

/**
 * @brief Устанавливает область интереса (ROI).
 * @param roi Изображение области интереса, которое нужно установить.
 */
void Image::setROI(cv::Mat& roi) {
    this->_roi_image.push_back(roi);
}

/**
 * @brief Предобрабатывает области интереса (ROI) для входа модели.
 * Конвертирует изображения в градации серого, изменяет их размер и нормализует пиксели.
 */
void Image::preprocessROI() {
    cv::Mat processed_image;

    if (_roi_image.size() > 0) {
        for (int i = 0; i < _roi_image.size(); i++) {
            // Конвертация в градации серого
            cv::Mat gray_image;
            cv::cvtColor(_roi_image[i], gray_image, cv::COLOR_BGR2GRAY);

            // Изменение размера ROI до входного размера модели
            cv::resize(gray_image, processed_image, cv::Size(48, 48));

            // Нормализация пикселей изображения от 0-255 до 0-1
            processed_image.convertTo(processed_image, CV_32FC3, 1.f / 255);

            // Добавление вектора входных данных модели
            _model_input_image.push_back(processed_image);
        }
    }
}