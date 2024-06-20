/**
 * @file Image.h
 * @brief Объявление класса Image.
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <iostream>

/**
 * @class Image
 * @brief Класс Image содержит полное изображение (кадр), области интереса (ROI) и изображение, готовое для ввода в модель.
 */
class Image {

public:
    /**
     * @brief Конструктор класса Image.
     */
    Image() {};

    /**
     * @brief Деструктор класса Image.
     */
    ~Image() {};

    /**
     * @brief Получает вектор областей интереса (ROI).
     * @return Вектор изображений областей интереса.
     */
    std::vector<cv::Mat> getROI();

    /**
     * @brief Устанавливает область интереса (ROI).
     * @param roi Изображение области интереса, которое нужно установить.
     */
    void setROI(cv::Mat& roi);

    /**
     * @brief Получает текущее изображение (кадр).
     * @return Текущее изображение.
     */
    cv::Mat getFrame();

    /**
     * @brief Устанавливает текущее изображение (кадр).
     * @param frame Изображение, которое нужно установить.
     */
    void setFrame(cv::Mat& frame);

    /**
     * @brief Предобрабатывает области интереса (ROI) для входа модели.
     * Конвертирует изображения в градации серого, изменяет их размер и нормализует пиксели.
     */
    void preprocessROI();

    /**
     * @brief Получает вектор изображений для входа модели.
     * @return Вектор изображений для входа модели.
     */
    std::vector<cv::Mat> getModelInput();

private:
    cv::Mat _frame; ///< Полное изображение (кадр).
    std::vector<cv::Mat> _roi_image; ///< Области интереса внутри рамки.
    std::vector<cv::Mat> _model_input_image; ///< Предобработанные изображения, готовые для ввода в модель.
};

#endif