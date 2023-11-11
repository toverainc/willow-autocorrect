FROM python:3.12.0-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY . .

EXPOSE 9000

CMD /app/entrypoint.sh