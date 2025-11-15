# Makefile
# Запуск: make <target>
# Примеры: make data, make train, make clean

.PHONY: all data split train clean help

# --- Настройки ---
PYTHON := python3
DATA_DIR := data
SPLIT_DIR := $(DATA_DIR)/splits
CHECKPOINT_DIR := checkpoints

# --- Цели ---

## Загрузить данные из облака
data:
	@echo "📥 Скачивание данных..."
	@$(PYTHON) tools/download_data.py
	@echo "✅ Данные сохранены в $(DATA_DIR)"

## Разделить на train/val
split:
	@echo "✂️  Разделение на train/val..."
	@$(PYTHON) tools/split_data.py
	@echo "✅ Разделение завершено: $(SPLIT_DIR)"

## Обучить модель (baseline)
train:
	@echo "🚀 Запуск обучения..."
	@mkdir -p $(CHECKPOINT_DIR)
	@$(PYTHON) experiments/train.py
	@echo "✅ Обучение завершено. Веса: $(CHECKPOINT_DIR)"

## Очистить производные данные (сохранить исходные)
clean:
	@echo "🧹 Очистка..."
	@rm -rf $(SPLIT_DIR) $(CHECKPOINT_DIR) logs/
	@echo "✅ Временные файлы удалены"

## Помощь
help:
	@echo "Доступные команды:"
	@echo "  make data    — скачать данные"
	@echo "  make split   — разделить на train/val"
	@echo "  make train   — обучить модель"
	@echo "  make clean   — удалить временные файлы"
	@echo "  make help    — эта справка"

## Запуск всех этапов
all: train