# Используем официальный образ Python
FROM python:3.12-slim

# Устанавливаем uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Копируем исходный код приложения
COPY . /app

RUN uv sync
# Команда для запуска приложения
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050"]
