# Makefile
# Запуск: make <target>
# Примеры: make download_data, make train, make clean

.PHONY: all download_data split_data train clean help

# --- Настройки ---
PYTHON := python3
CHECKPOINTS_DIR := checkpoints
LOGS_GIR := logs


DATA_FILE := data.zip
DATA_DIR := data
PREDICT_DATA_FILE := predict_data.zip
PREDICT_DATA_DIR := predict_images
IMAGES_DIR := images
MASKES_DIR := masks
SPLIT_DIR ?= splits
TRAIN_SIZE ?= 
VAL_SIZE ?= 
MODEL_NAME ?= 


# --- Цели ---

## Загрузить данные из облака
download_data:
	@echo "📥 Скачивание данных..."
# 	@$(PYTHON) tools/download_data.py \
# 		--data_file $(DATA_FILE) \
# 		--predict_data_file $(PREDICT_DATA_FILE)
	
	@echo "Распаковка..."
	@unzip -q $(DATA_FILE) -d $(DATA_DIR)

	@unzip -q $(PREDICT_DATA_FILE) -d $(PREDICT_DATA_DIR)
	@mv $(PREDICT_DATA_DIR)/predict_input $(DATA_DIR)/$(PREDICT_DATA_DIR)

	@mv $(DATA_DIR)/input $(DATA_DIR)/$(IMAGES_DIR)
	@mv $(DATA_DIR)/target $(DATA_DIR)/$(MASKES_DIR)

# 	@rm $(DATA_FILE)
# 	@rm $(PREDICT_DATA_FILE)
	
	@echo "✅ Данные сохранены в $(DATA_DIR)"

## Разделить на train/val
# make split_data SPLIT_DIR=splits_80_20 TRAIN_DIR=80 VAL_DIR=20
split_data:
	@echo "✂️  Разделение на train/val..."
	python tools/split_data.py \
		--split_dir "$(SPLIT_DIR)" \
		$(if $(TRAIN_SIZE),--train_size $(TRAIN_SIZE)) \
		$(if $(VAL_SIZE),--val_size $(VAL_SIZE))
	@echo "✅ Разделение завершено: $(DATA_DIR)/$(SPLIT_DIR)"

## Обучить модель
# make train SPLIT_DIR=splits_80_20
train:
	@echo "🚀 Запуск обучения..."
	@mkdir -p $(CHECKPOINTS_DIR)
	@$(PYTHON) experiments/train.py \
		--split_dir "$(SPLIT_DIR)"
	@echo "✅ Обучение завершено. Веса: $(CHECKPOINTS_DIR)"

# Сделать предсказание на тестовых данных
predict:
	@echo "🚀 Запуск обучения..."
	@$(PYTHON) experiments/predict.py \
		--predict_dir "$(PREDICT_DATA_DIR)" \
		--model_name "$(MODEL_NAME)"

	@cd results && zip -q predict_target.zip predict_target/*.png && cd ..
	@echo "✅ Прогнозирование завершено."

## Очистить производные данные (сохранить исходные)
clean_splits:
	@echo "🧹 Очистка..."

	@for dir in data/*/; do \
		if [ "$$(basename "$$dir")" != $(IMAGES_DIR) ] && [ "$$(basename "$$dir")" != $(MASKES_DIR) ]; then \
			echo "Удаление $$dir"; \
			rm -rf "$$dir"; \
		fi; \
	done

	@echo "✅ Временные файлы удалены"

# Очистить чекпоинты
clean_checkpoints:
	@if [ -d $(CHECKPOINTS_DIR) ]; then \
		echo "Удаление $(CHECKPOINTS_DIR)..."; \
		rm -rf $(CHECKPOINTS_DIR); \
	fi

	@echo "✅ Временные файлы удалены"

# Очистить логи
clean_logs:
	@if [ -d $(LOGS_GIR) ]; then \
		echo "Удаление $(LOGS_GIR)..."; \
		rm -rf $(LOGS_GIR); \
	fi

	@echo "✅ Временные файлы удалены"

## Помощь
help:
	@echo "Доступные команды:"
	@echo "  make download_data    — скачать данные"
	@echo "  make split_data   — разделить на train/val"
	@echo "  make train   — обучить модель"
	@echo "  make clean   — удалить временные файлы"
	@echo "  make help    — эта справка"

## Запуск всех этапов
all: train