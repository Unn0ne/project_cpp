#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/opencv.hpp>

class Video {
public:
    // Конструктор принимает объект cv::VideoCapture
    Video(cv::VideoCapture& capture);

    // Метод для получения длины видео в секундах
    double getLengthInSeconds() const;

    // Оператор для доступа к кадру в заданную секунду
    cv::Mat operator[](double seconds) const;

    cv::Mat getFrame(int frameNumber) const;

    bool saveFrame(int frameNumber, const std::string& filename) const;

private:
    cv::VideoCapture& capture;
    double fps;  // Количество кадров в секунду
    double lengthInSeconds;  // Длина видео в секундах
};

#endif // VIDEO_H

