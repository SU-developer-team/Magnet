import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Путь к вашей модели TFLite
MODEL_PATH = 'runs/train/exp27/weights/best-fp16.tflite'

# Загрузка модели TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Получение деталей входа и выхода модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Получение размеров входного изображения модели
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Функция для предобработки изображения
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    image_np = np.array(image).astype(np.float32)
    # Нормализация изображения (зависит от вашей модели)
    image_np /= 255.0
    # Добавление измерения батча
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

# Функция для постобработки результатов
def postprocess(outputs, conf_threshold=0.25, iou_threshold=0.45):
    # Предполагается, что модель возвращает [boxes, classes, scores]
    # Необходимо адаптировать в соответствии с выходами вашей модели
    boxes, classes, scores = outputs
    # Фильтрация по порогу уверенности
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    classes = classes[mask]
    scores = scores[mask]

    # Применение NMS (не обязательно, если модель уже применяет NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        boxes = boxes[indices.flatten()]
        classes = classes[indices.flatten()]
        scores = scores[indices.flatten()]
    return boxes, classes, scores

# Функция для выполнения инференса
def run_inference(image_path):
    # Предобработка изображения
    input_data = preprocess_image(image_path)

    # Установка входных данных
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Выполнение инференса
    interpreter.invoke()

    # Получение выходных данных
    outputs = []
    for output_detail in output_details:
        output = interpreter.get_tensor(output_detail['index'])
        outputs.append(output)
    
    # Постобработка результатов
    boxes, classes, scores = postprocess(outputs)

    return boxes, classes, scores

# Функция для отображения результатов на изображении
def display_results(image_path, boxes, classes, scores):
    image = cv2.imread(image_path)
    for box, cls, score in zip(boxes, classes, scores):
        # Координаты бокса
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (int(xmin * image.shape[1]), int(xmax * image.shape[1]),
                                      int(ymin * image.shape[0]), int(ymax * image.shape[0]))
        # Рисование прямоугольника
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Текст с классом и уверенностью
        label = f'Class {int(cls)}: {score:.2f}'
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    # Отображение изображения
    cv2.imshow('Inference', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Использование: python infer.py <путь_к_изображению>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    boxes, classes, scores = run_inference(image_path)
    display_results(image_path, boxes, classes, scores)
