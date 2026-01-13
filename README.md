# Trader MVP Platform

Многоcервисный MVP для трейдинга: слои Market, News и ML, сервисы для сигналов и paper-трейдинга, всё поднимается через Docker Compose.

## Что внутри
- Python 3.11, FastAPI, Pydantic v2, Redis Streams/PubSub, Postgres.
- Сервисы: market_service, pattern_service, news_service, ai_nlp_service, trader_service, trainer_service, telegram_bot_service.
- Общие DTO и конфиги в `src/shared`.
- ORM: SQLAlchemy async, базовые таблицы под market/news/ML/trades/bot_state.

## Быстрый старт
1. Установить Python 3.11+.
2. Создать `.env` на основе `env.example`.
3. Применить миграции БД:
   ```bash
   alembic upgrade head
   ```
4. Запустить инфраструктуру и сервисы (Docker) или локально:
   ```bash
   docker compose up --build
   # либо локально:
   uvicorn market_service.main:app --reload
   ```

## Сервисы (кратко)
- `market_service`: WebSocket Bybit (позже), хранит последнее состояние, считает базовые фичи, публикует события.
- `pattern_service`: превращает фичи в сетапы (trend / breakout / mean-revert) и market_bias.
- `news_service`: собирает новости/RSS, дедуп, пишет в Postgres.
- `ai_nlp_service`: правила + (опционально) LLM, выдаёт NewsScore.
- `trader_service`: объединяет Market+News+ML, риск-проверки, paper/demo сделки.
- `trainer_service`: джоба для подготовки датасетов и обучения модели.
- `telegram_bot_service`: статус/пауза/возобновление/kill через HTTP (дальше — Telegram).

## Структура кода
- `src/shared` — общие конфиги, логирование, схемы событий/DTO.
- `src/shared/models.py` — SQLAlchemy ORM таблицы (market_bars, market_features, pattern_signals, news_events, news_scores, model_predictions, decisions, trade_plans, trades, bot_state).
- `src/<service>` — код конкретного сервиса. Каждый сервис — небольшой FastAPI-приложение + фоновый воркер.

## Следующие шаги
- Добавить реальные клиенты (Bybit WS, RSS, Postgres/Redis).
- Реализовать фичи/сетапы и агрегатор сигналов.
- Прописать миграции (Alembic) и метрики (Prometheus/Grafana).

