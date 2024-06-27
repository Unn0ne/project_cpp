
/**
 * @file main.cpp
 * @brief Главный файл программы для распознавания эмоций на лицах в реальном времени.
 */

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>
#include <iomanip>

#include "FaceDetector.h"
#include "Image.h"
#include "Model.h"
#include "Video.h"

#ifndef TEST_ENV


/**
 * @brief Путь к модели для детекции лиц с использованием каскадных классификаторов.
 */
const std::string FACE_DETECTOR_MODEL_PATH = "../model/haarcascade_frontalface_alt2.xml";
/**
 * @brief Путь к модели TensorFlow для предсказания эмоций.
 */
const std::string TENSORFLOW_MODEL_PATH = "../model/tensorflow_model.pb";
/**
 * @brief Название приложения.
 */
const std::string APP_NAME = "Real-Time Facial Emotion Recognition";
/**
 * @brief Путь к файлу.
 */
const std::string WAY = "";

// Вывод гистограммы частот 
void printHistogram(const std::vector<std::string>& words) {
    // Инициализация словаря для хранения частот
    std::unordered_map<std::string, int> frequency = {
        {"Angry", 0},
        {"Disgust", 0},
        {"Fear", 0},
        {"Happy", 0},
        {"Sad", 0},
        {"Surprise", 0},
        {"Neutral", 0}
    };

    // Подсчет частоты слов
    for (const std::string& word : words) {
        if (frequency.find(word) != frequency.end()) {
            frequency[word]++;
        }
    }

    // Вывод гистограммы
    for (const auto& pair : frequency) {
        std::cout << std::setw(10) << pair.first << " : " << std::string(pair.second, '*') << " (" << pair.second << ")\n";
    }
}

// Сохранение гистограммы в текстовый файл
void saveHistogramToFile(const std::vector<std::string>& words, const std::string& filename) {
    std::unordered_map<std::string, int> frequency = {
        {"Angry", 0},
        {"Disgust", 0},
        {"Fear", 0},
        {"Happy", 0},
        {"Sad", 0},
        {"Surprise", 0},
        {"Neutral", 0}
    };

    for (const std::string& word : words) {
        if (frequency.find(word) != frequency.end()) {
            frequency[word]++;
        }
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& pair : frequency) {
            file << pair.first << " : " << std::string(pair.second, '*') << " (" << pair.second << ")\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Построение графика эмоций по времени
void plotEmotionGraph(const std::vector<std::string>& emotions) {
    std::unordered_map<std::string, int> emotionMap = {
        {"Angry", 0},
        {"Disgust", 1},
        {"Fear", 2},
        {"Happy", 3},
        {"Sad", 4},
        {"Surprise", 5},
        {"Neutral", 6}
    };

    std::vector<int> time;
    std::vector<int> emotionValues;

    for (int i = 0; i < emotions.size(); ++i) {
        time.push_back(i);
        emotionValues.push_back(emotionMap[emotions[i]]);
    }

    cv::Mat plot(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    int x_scale = plot.cols / (time.size() + 1);
    int y_scale = plot.rows / (emotionMap.size() + 1);

    for (int i = 1; i < time.size(); ++i) {
        cv::line(plot, cv::Point((i-1) * x_scale, plot.rows - emotionValues[i-1] * y_scale),
                 cv::Point(i * x_scale, plot.rows - emotionValues[i] * y_scale),
                 cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Emotion Graph", plot);
    cv::waitKey(0);
}

/**
 * @brief Главная функция программы.
 * @return Код завершения программы.
 */
int main()
{
    // Инициализация видеокадра, который будет считываться с камеры
    // Инициализация всех необходимых объектов
    int anser{0};
    std::cout << "What you want to use" << std::endl
              << "0) image" << std::endl
              << "1) camera" << std::endl
              << "2) video" << std::endl;
    std::cin >> anser;

    if (!anser) {
        Model model(TENSORFLOW_MODEL_PATH);
        FaceDetector face_detector;
        Image image_and_ROI;
        std::cout << "input name of the image like <name.jpg>" << std::endl;
        cv::Mat frame;
        std::string noname;
        std::cin >> noname;
        frame = cv::imread("../src/" + noname);

        // Создание окна с названием приложения
        cv::namedWindow(APP_NAME);

        // Главный цикл программы
        // Выполнение детекции лиц и рисование рамок
        face_detector.detectFace(frame);

        // Рисование рамок на изображении
        image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

        // Получение областей интереса (ROI)
        std::vector<cv::Mat> roi_image = image_and_ROI.getROI();

        if (roi_image.size() > 0) {
            // Предобработка изображения для модели
            image_and_ROI.preprocessROI();
            // Выполнение предсказания
            std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
            std::string emotion_prediction_2 = model.ans(image_and_ROI);

            // Добавление текста предсказания на изображение
            image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
            std::cout << "it should be " << emotion_prediction[0] << std::endl
                      << "also could be " << emotion_prediction_2 << std::endl;
        }

        cv::Mat output_frame = image_and_ROI.getFrame();

        if (!output_frame.empty()) {
            // Отображение видеокадра в окне
            imshow(APP_NAME, output_frame);
        } else {
            // Если выходной кадр пуст (например, детектор лиц не обнаружил ничего), просто отображаем оригинальное видео
            imshow(APP_NAME, frame);
        }

        // Ожидание нажатия любой клавиши в течение 10 мс.
        // Если нажата клавиша 'Esc', выход из программы
        if (cv::waitKey(0) == 27) {
            std::cout << "Esc key is pressed by user. Stopping the program" << std::endl;
            return 0;
        }

        return 0;
    } if(anser == 1){
        cv::Mat frame;
        // Инициализация всех необходимых объектов
        Model model(TENSORFLOW_MODEL_PATH);
        FaceDetector face_detector;
        Image image_and_ROI;

        // Инициализация объекта захвата видео с использованием камеры по умолчанию
        cv::VideoCapture cap(0);
        // Создание окна с названием приложения
        cv::namedWindow(APP_NAME);

        // Главный цикл программы
        while (true) {
            // Считывание нового кадра с видео
            bool bSuccess = cap.read(frame);
// Прерывание цикла, если не удается захватить кадры
            if (!bSuccess) {
                std::cout << "Video camera is disconnected. Stopping the program" << std::endl;
                std::cin.get(); // Ожидание нажатия любой клавиши
                break;
            }

            // Выполнение детекции лиц и рисование рамок
            face_detector.detectFace(frame);

            // Рисование рамок на изображении
            image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

            // Получение областей интереса (ROI)
            std::vector<cv::Mat> roi_image = image_and_ROI.getROI();

            if (roi_image.size() > 0) {
                // Предобработка изображения для модели
                image_and_ROI.preprocessROI();
                // Выполнение предсказания
                std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
                // Добавление текста предсказания на изображение
                image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
            }

            cv::Mat output_frame = image_and_ROI.getFrame();

            if (!output_frame.empty()) {
                // Отображение видеокадра в окне
                imshow(APP_NAME, output_frame);
            } else {
                // Если выходной кадр пуст (например, детектор лиц не обнаружил ничего), просто отображаем оригинальное видео
                imshow(APP_NAME, frame);
            }

            // Ожидание нажатия любой клавиши в течение 10 мс.
            // Если нажата клавиша 'Esc', выход из программы
            if (cv::waitKey(100) == 27) {
                std::cout << "Esc key is pressed by user. Stopping the program" << std::endl;
                break;
            }
        }

        return 0;
    }
    if (anser == 2) {

      std::cout<<"input name of the video: ";
      std::string name{""};
      std::cin>>name;
      cv::VideoCapture cap("../src/" + name);

        Video mp(cap);
        std::vector<std::string> spectrum;

        for(double time = 0; time < mp.getLengthInSeconds(); time = time + 1.0) {
          cv::Mat frame = mp[time];
          // Инициализация всех необходимых объектов
          Model model(TENSORFLOW_MODEL_PATH);
          FaceDetector face_detector;
          Image image_and_ROI;
          std::string nnn{};
          // Выполнение детекции лиц и рисование рамок
          face_detector.detectFace(frame);

          // Рисование рамок на изображении
          image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

          // Получение областей интереса (ROI)
          std::vector<cv::Mat> roi_image = image_and_ROI.getROI();

          if (roi_image.size() > 0) {
              // Предобработка изображения для модели
              image_and_ROI.preprocessROI();
              // Выполнение предсказания
              std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
              // Добавление текста предсказания на изображение
              image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);

              for(auto a : emotion_prediction[0]) {

                if(a == ':'){
                  break;
                }
                nnn = nnn + a;

              }
              std::cout<<emotion_prediction[0]<<std::endl;
              spectrum.push_back(nnn);
          }
        }
      std::cout<<"Histogram of frequency"<<std::endl;
      printHistogram(spectrum);
      

      std::string histogram_filename = "emotion_histogram.txt";
        saveHistogramToFile(spectrum, histogram_filename);
        std::cout << "Histogram saved to " << histogram_filename << std::endl;

        plotEmotionGraph(spectrum);


    }
    return 0;
}

#endif
