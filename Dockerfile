FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg DEBIAN_FRONTEND=noninteractive

# deps di sistema + unixODBC
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates gnupg apt-transport-https build-essential \
      unixodbc unixodbc-dev && \
    rm -rf /var/lib/apt/lists/*

# repo Microsoft (Debian 12 / bookworm) + ODBC 18
RUN set -eux; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft-prod.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh","-c","uvicorn main_script:app --host 0.0.0.0 --port ${PORT:-8000}"]
