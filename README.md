# Студенческий хакатон от Роснефти

## 🚀 Быстрый старт

### 1. Настройка окружения
```bash
# Создать и активировать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Загрузка и подготовка данных
```bash
# Скачать данные из облака и разделить на train/val
make download_data
make split_data SPLIT_DIR=splits_80_20 TRAIN_SIZE=80 VAL_SIZE=20
```

### 3. Обучение модели
```bash
# Запустить обучение
make train SPLIT_DIR=splits_80_20
```

### 4. Предсказание на тестовых данных
Нужно создать папку `saved_models` и положить туда обученную модель (например, `checkpoint_epoch_5.pth`). Затем нужно указать эту модель для прогнозирования.
```bash
# Запустить прогнозирование
make predict MODEL_NAME=checkpoint_epoch_5
```
