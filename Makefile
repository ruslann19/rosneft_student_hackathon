# Makefile
# Запуск: make <target>
# Примеры: make download_data, make train, make clean

.PHONY: all download_data split_data train clean help

# --- Настройки ---
PYTHON := python3
CHECKPOINT_DIR := checkpoints

ZIP_FILE := data.zip
DATA_DIR := data
IMAGES_DIR := images
MASKES_DIR := maskes
SPLIT_DIR ?= splits
TRAIN_SIZE ?= 
VAL_SIZE ?= 


# --- Цели ---

## Загрузить данные из облака
download_data:
	@echo "📥 Скачивание данных..."
	@$(PYTHON) tools/download_data.py --zip_file $(ZIP_FILE)
	
	@echo "Распаковка..."
	@unzip -q $(ZIP_FILE) -d $(DATA_DIR)

	@rm $(ZIP_FILE)

	@mv $(DATA_DIR)/input $(DATA_DIR)/$(IMAGES_DIR)
	@mv $(DATA_DIR)/target $(DATA_DIR)/$(MASKES_DIR)
	
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
	@mkdir -p $(CHECKPOINT_DIR)
	@$(PYTHON) experiments/train.py --split_dir "$(SPLIT_DIR)"
	@echo "✅ Обучение завершено. Веса: $(CHECKPOINT_DIR)"

## Очистить производные данные (сохранить исходные)
clean:
	@echo "🧹 Очистка..."

	@for dir in data/*/; do \
		if [ "$$(basename "$$dir")" != $(IMAGES_DIR) ] && [ "$$(basename "$$dir")" != $(MASKES_DIR) ]; then \
			echo "Удаление $$dir"; \
			rm -rf "$$dir"; \
		fi; \
	done

	@if [ -d $(CHECKPOINT_DIR) ]; then \
		echo "Удаление $(CHECKPOINT_DIR)..."; \
		rm -rf $(CHECKPOINT_DIR); \
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