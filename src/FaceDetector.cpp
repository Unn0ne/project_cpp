/**
 * @file FaceDetector.cpp
 * @brief Реализация методов класса FaceDetector.
 */

#include <opencv2/opencv.hpp>
#include "FaceDetector.h"
#include "Image.h"

/**
 * @brief Конструктор класса FaceDetector.
 * Загружает каскадный классификатор.
 */
FaceDetector::FaceDetector() {
    // Загрузка каскадного классификатора
    cascade.load(FACE_DETECTOR_MODEL_PATH);
}

/**
 * @brief Обнаружение лиц на изображении.
 * @param frame Изображение, на котором нужно обнаружить лица.
 */
void FaceDetector::detectFace(cv::Mat& frame) {
    cv::Mat gray_img;
    cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray_img, gray_img);

    // Обнаружение лиц
    cascade.detectMultiScale(gray_img, this->faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100));
}

/**
 * @brief Рисует рамку вокруг обнаруженных лиц на изображении.
 * @param frame Изображение, на котором нужно нарисовать рамку.
 * @return Изображение с нанесенными рамками и областью интереса (ROI).
 */
Image FaceDetector::drawBoundingBoxOnFrame(cv::Mat& frame) {
    Image image_and_ROI;

    // Для каждого обнаруженного лица рисуется рамка
    if (faces.size() > 0) {
        for (int i = 0; i < faces.size(); i++) {
            cv::Rect r = faces[i];

            // Рисование прямоугольника вокруг лица
            rectangle(frame,
                      cv::Point(r.x, r.y),
                      cv::Point(r.x + r.width, r.y + r.height),
                      cv::Scalar(255, 0, 0), 3, 8, 0);

            cv::Rect roi_coord(r.x, r.y, r.width, r.height);
            cv::Mat roi_image = frame(roi_coord);

            image_and_ROI.setROI(roi_image);
            image_and_ROI.setFrame(frame);
        }
    }

    return image_and_ROI;
}

/**
 * @brief Печатает текст с предсказанием эмоций на изображении.
 * @param image_and_ROI Изображение с рамками вокруг лиц.
 * @param emotion_prediction Вектор строк с предсказанными эмоциями.
 * @return Изображение с текстом предсказаний.
 */
Image FaceDetector::printPredictionTextToFrame(Image& image_and_ROI, std::vector<std::string>& emotion_prediction) {
    cv::Mat img = image_and_ROI.getFrame();

    if (faces.size() > 0) {
        for (int i = 0; i < faces.size(); i++) {
            cv::Rect r = faces[i];

            // Написание текста с предсказанием на рамке
            cv::putText(img, // целевое изображение
                        emotion_prediction[i], // текст - результат работы модели
                        cv::Point(r.x, r.y - 10), // верхняя левая позиция рамки
                        cv::FONT_HERSHEY_DUPLEX,
                        1.0,
                        CV_RGB(118, 185, 0), // цвет шрифта
                        2);
        }
    }

    image_and_ROI.setFrame(img);

    return image_and_ROI;
}