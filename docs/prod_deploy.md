# Prod deploy checklist

- Postgres: managed (RDS/CloudSQL/Supabase) with daily backups; restrict network to app hosts; env `POSTGRES_*` point to prod host; enable WAL retention and monitoring.
- Redis: managed (AWS Elasticache/Upstash/Redis Cloud) with persistence if требуется; network-restricted.
- Secrets: `.env` только локально; в прод — переменные окружения/secret manager; отдельные токены для prod.
- Monitoring: Prometheus/Grafana или облачный APM; алерты в Telegram по ошибкам, halt, дневному лимиту.
- CI/CD: GitHub Actions (`.github/workflows/ci.yml`) → build/test; деплой контейнеров в registry и на хост/оркестр (docker compose/k8s).
- Storage: `data/models` в Object Storage (S3/MinIO) для артефактов моделей; sync на старте.
- TLS/Ingress: если нужен webhook — HTTPS через reverse proxy (nginx/traefik) или облачный LB.
