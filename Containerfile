# Используем официальный образ Python
FROM python:3.12-slim

# Устанавливаем системные зависимости для сборки C-расширения
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Копируем исходный код приложения
COPY . /app

RUN uv sync
# Проверяем, что native-ускорение lesson2 доступно
RUN uv run python -c "import lesson2_rhs; print('lesson2 native:', lesson2_rhs.native_available())"
# Команда для запуска приложения
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050"]
