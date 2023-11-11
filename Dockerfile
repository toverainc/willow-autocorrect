ARG TYPESENSE_VER="0.25.1"

FROM python:3.12.0-slim-bookworm
ARG TYPESENSE_VER

# Install typesense-server with md5sum workaround
ARG BUILDARCH

RUN apt-get update && apt-get install --no-install-recommends -y coreutils curl && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/sbin
RUN curl -o typesense-server.tar.gz https://dl.typesense.org/releases/${TYPESENSE_VER}/typesense-server-${TYPESENSE_VER}-linux-${BUILDARCH}.tar.gz && \
    tar xf typesense-server.tar.gz && echo -n '  typesense-server' >> typesense-server.md5.txt \
    md5sum -c typesense-server.md5.txt && rm typesense-server*.*

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY . .

EXPOSE 9000

CMD /app/entrypoint.sh