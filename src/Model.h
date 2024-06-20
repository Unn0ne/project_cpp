/**
 * @file Model.h
 * @brief Объявление класса Model.
 */

#ifndef MODEL_H
#define MODEL_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Image.h"

/**
 * @class Model
 * @brief Класс Model содержит код для загрузки предобученной модели TensorFlow и позволяет делать предсказания на новых изображениях.
 */
class Model {
public:
    /**
     * @brief Конструктор загружает предобученную модель TensorFlow и инициализирует отображение ID классов в строковые метки (например, happy, angry, sad и т.д.).
     * @param model_filename Путь к файлу модели (.pb файл содержит всё необходимое о модели).
     */
    Model(const std::string& model_filename);

    /**
     * @brief Деструктор класса Model.
     */
    ~Model() {};

    /**
     * @brief Функция предсказания модели, принимает изображение на вход и возвращает предсказанную метку и вероятность.
     * @param image Изображение для предсказания.
     * @return Вектор строк с предсказанными эмоциями и их вероятностями.
     */
    std::vector<std::string> predict(Image& image);

    /**
     * @brief Функция предсказания модели, принимает изображение на вход и возвращает первую предсказанную эмоцию.
     * @param image Изображение для предсказания.
     * @return Строка с первой предсказанной эмоцией и её вероятностью.
     */
    std::string ans(Image& image);

private:
    cv::dnn::Net network; ///< Нейронная сеть модели.
    std::map<int, std::string> classid_to_string; ///< Отображение ID класса в строковую метку.
};

#endif