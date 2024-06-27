#include "Video.h"
#include <opencv2/opencv.hpp>

Video::Video(cv::VideoCapture& capture) : capture(capture) {
    // Получаем количество кадров в секунду
    fps = capture.get(cv::CAP_PROP_FPS);

    // Получаем общее количество кадров и вычисляем длину видео в секундах
    double frameCount = capture.get(cv::CAP_PROP_FRAME_COUNT);
    lengthInSeconds = frameCount / fps;
}

// Выводим длину видео 
double Video::getLengthInSeconds() const {
    return lengthInSeconds;
}

cv::Mat Video::operator[](double seconds) const {
    // Вычисляем номер кадра на основе заданного времени в секундах
    int frameNumber = static_cast<int>(seconds * fps);

    // Устанавливаем положение кадра
    capture.set(cv::CAP_PROP_POS_FRAMES, frameNumber);

    // Читаем и возвращаем кадр
    cv::Mat frame;
    capture.read(frame);
    return frame;
}
cv::Mat Video::getFrame(int frameNumber) const {
    // Устанавливаем положение кадра
    capture.set(cv::CAP_PROP_POS_FRAMES, frameNumber);

    // Читаем и возвращаем кадр
    cv::Mat frame;
    capture.read(frame);
    return frame;
}

bool Video::saveFrame(int frameNumber, const std::string& filename) const {
    // Получаем кадр по номеру
    cv::Mat frame = getFrame(frameNumber);

    // Проверяем, пустой ли кадр
    if (frame.empty()) {
        return false;
    }

    // Сохраняем кадр в файл
    return cv::imwrite(filename, frame);
}


