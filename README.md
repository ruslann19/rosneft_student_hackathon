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
make data
make split
```

### 3. Обучение модели
```bash
# Запустить обучение
make train
```
