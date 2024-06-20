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

#include "FaceDetector.h"
#include "Image.h"
#include "Model.h"

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
<<<<<<< HEAD
/** 
 * @brief Путь к файлу.
 */
const std::string WAY = "";

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
              << "1) camera" << std::endl;
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
=======
const std::string WAY = "";
int main()
{
    // Initialise video frame that will be read from the camera
    // Initialise all the required objects
    int anser{0};
    std::cout<<"What you want to use"<<std::endl 
             <<"0) image"<<std::endl 
             <<"1) camera"<<std::endl;
    std::cin>>anser;

    if(!anser) {
    
    Model model(TENSORFLOW_MODEL_PATH);
    FaceDetector face_detector;
    Image image_and_ROI;
    std::cout<<"input name of the image like <name.jpg>"<<std::endl;
    cv::Mat frame;
    std::string noname; 
    std::cin>>noname;
    frame = cv::imread("../src/" + noname);
    //cv::Mat frame;
    //frame = cv::imread("/Users/makariusiii/CppND-Facial-Emotion-Recognition/src/image.jpg");
    //if (frame.empty()):
    //{
    //   std::cout << "Unable to read image" << std::endl;
    //    return -1;

    //}
    
    // Initialise video capture object with default camera
    //create a window with the window name
    cv::namedWindow(APP_NAME);
>>>>>>> refs/remotes/origin/main

        // Создание окна с названием приложения
        cv::namedWindow(APP_NAME);

        // Главный цикл программы
        // Выполнение детекции лиц и рисование рамок
        face_detector.detectFace(frame);

        // Рисование рамок на изображении
        image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);
<<<<<<< HEAD

        // Получение областей интереса (ROI)
=======
        // Get Image ROIs

       // std::cout<<model.ans(image_and_ROI);
>>>>>>> refs/remotes/origin/main
        std::vector<cv::Mat> roi_image = image_and_ROI.getROI();
    
        if (roi_image.size() > 0) {
            // Предобработка изображения для модели
            image_and_ROI.preprocessROI();
            // Выполнение предсказания
            std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
            std::string emotion_prediction_2 = model.ans(image_and_ROI);

<<<<<<< HEAD
            // Добавление текста предсказания на изображение
            image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
            std::cout << "it should be " << emotion_prediction[0] << std::endl
                      << "also could be " << emotion_prediction_2 << std::endl;
=======
            // Add prediction text to the output video frame
            image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
            std::cout<<"it should be "<<emotion_prediction[0]<<std::endl
                     <<"also could be "<<emotion_prediction_2<<std::endl;
                   

>>>>>>> refs/remotes/origin/main
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

<<<<<<< HEAD
        return 0;
    } else {
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
=======
    
    return 0;
    }

    else {
    cv::Mat frame;
    // Initialise all the required objects
    Model model(TENSORFLOW_MODEL_PATH);
    FaceDetector face_detector;
    Image image_and_ROI;

    // Initialise video capture object with default camera
    cv::VideoCapture cap(0);	
    //create a window with the window name
    cv::namedWindow(APP_NAME);

    // Main Program Loop    
    while (true)
    {
        // read a new frame from video 
        bool bSuccess = cap.read(frame); 

        // break the while loop if the frames cannot be captured
        if (bSuccess == false) 
        {
            std::cout << "Video camera is disconnected. Stopping the program" << std::endl;
            std::cin.get(); //Wait for any key press
            break;
        }

        //Run Face Detection and draw bounding box
        face_detector.detectFace(frame);

        // Draw bounding box to frame
        image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

        // Get Image ROIs
        std::vector<cv::Mat> roi_image = image_and_ROI.getROI();
    
        if (roi_image.size()>0) {
            // Preprocess image ready for model
            image_and_ROI.preprocessROI();
            // Make Prediction
            std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
            // Add prediction text to the output video frame
            image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
        }

        cv::Mat output_frame = image_and_ROI.getFrame();

        if (!output_frame.empty()) {
            // Display the video frame to the window
            imshow (APP_NAME, output_frame);
        } else {
            // if the output frame is empty (ie. the facedetector didn't detect anything), just display the original video capture
            imshow (APP_NAME, frame);
        }

        // wait for for 10 ms until any key is pressed.  
        // if the 'Esc' key is pressed, break the program loop
        if (cv::waitKey(100) == 27)
        {
            std::cout << "Esc key is pressed by user. Stopping the program" << std::endl;
            break;
        }

    }
    
    return 0;
>>>>>>> refs/remotes/origin/main
    }
    return 0;
}
