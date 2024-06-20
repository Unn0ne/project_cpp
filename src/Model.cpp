/**
 * @file Model.cpp
 * @brief Реализация методов класса Model.
 */

#include <opencv2/opencv.hpp>
#include "Model.h"

/**
 * @brief Конструктор класса Model.
 * @param model_filename Путь к файлу модели TensorFlow.
 */
Model::Model(const std::string& model_filename) 
    : network(cv::dnn::readNet(model_filename)), // Загрузка модели TensorFlow
      classid_to_string({{0, "Angry"}, 
                         {1, "Disgust"}, 
                         {2, "Fear"}, 
                         {3, "Happy"}, 
                         {4, "Sad"}, 
                         {5, "Surprise"}, 
                         {6, "Neutral"}}) // Создание отображения от ID класса к меткам классов
{}

/**
 * @brief Выполняет предсказание эмоций на основе входного изображения.
 * @param image Изображение для предсказания.
 * @return Вектор строк с предсказанными эмоциями и их вероятностями.
 */
std::vector<std::string> Model::predict(Image& image) {
    // Извлечение изображений областей интереса (ROI) для входа в модель
    std::vector<cv::Mat> roi_image = image.getModelInput();
    std::vector<std::string> emotion_prediction;

    if (roi_image.size() > 0) { 
        for (int i = 0; i < roi_image.size(); i++) {
            // Конвертация в blob
            cv::Mat blob = cv::dnn::blobFromImage(roi_image[i]);

            // Передача blob в сеть
            this->network.setInput(blob);

            // Прямой проход по сети
            cv::Mat prob = this->network.forward();

            // Сортировка вероятностей и индексов
            cv::Mat sorted_probabilities;
            cv::Mat sorted_ids;
            cv::sort(prob.reshape(1, 1), sorted_probabilities, cv::SORT_DESCENDING);
            cv::sortIdx(prob.reshape(1, 1), sorted_ids, cv::SORT_DESCENDING);

            // Получение наивысшей вероятности и соответствующего ID класса
            float top_probability = sorted_probabilities.at<float>(0);
            int top_class_id = sorted_ids.at<int>(0);

            // Отображение ID класса на название класса (например, happy, sad, angry, disgust и т.д.)
            std::string class_name = this->classid_to_string.at(top_class_id);

            // Строка с результатом предсказания для вывода
            std::string result_string = class_name + ": " + std::to_string(top_probability * 100) + "%";

            // Добавление результата в вектор предсказаний
            emotion_prediction.push_back(result_string);
        }
    }

    return emotion_prediction;
}

/**
 * @brief Выполняет предсказание эмоций на основе входного изображения и возвращает первую эмоцию.
 * @param image Изображение для предсказания.
 * @return Строка с первой предсказанной эмоцией и её вероятностью.
 */
std::string Model::ans(Image& image) {
    // Извлечение изображений областей интереса (ROI) для входа в модель
    std::vector<cv::Mat> roi_image = image.getModelInput();
    std::vector<std::string> emotion_prediction;

    if (roi_image.size() > 0) { 
        for (int i = 0; i < roi_image.size(); i++) {
            // Конвертация в blob
            cv::Mat blob = cv::dnn::blobFromImage(roi_image[i]);

            // Передача blob в сеть
            this->network.setInput(blob);

            // Прямой проход по сети
            cv::Mat prob = this->network.forward();

            // Сортировка вероятностей и индексов
            cv::Mat sorted_probabilities;
            cv::Mat sorted_ids;
            cv::sort(prob.reshape(1, 1), sorted_probabilities, cv::SORT_DESCENDING);
            cv::sortIdx(prob.reshape(1, 1), sorted_ids, cv::SORT_DESCENDING);

            // Получение наивысшей вероятности и соответствующего ID класса
            float top_probability = sorted_probabilities.at<float>(0);
            int top_class_id = sorted_ids.at<int>(0);

            // Отображение ID класса на название класса (например, happy, sad, angry, disgust и т.д.)
            std::string class_name = this->classid_to_string.at(top_class_id);

            // Строка с результатом предсказания для вывода
            std::string result_string = class_name + ": " + std::to_string(top_probability * 100) + "%";

            // Добавление результата в вектор предсказаний
            emotion_prediction.push_back(result_string);
        }
    }

    return emotion_prediction.size() > 0 ? emotion_prediction[0] : "";
}