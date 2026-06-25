FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=9005
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install git for cloning
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Clone and install AWARE package
WORKDIR /app
RUN git clone https://github.com/oscarboyle/aware-music.git aware44k && \
    cd aware44k && pip install --no-cache-dir -e .

# Go back to app directory
WORKDIR /app

# Install service-specific dependencies
COPY /src/plugins/models/aware44k/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/plugins/models/aware44k/app.py .
COPY src/plugins/models/aware44k/model.py .
COPY src/utils ./utils

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT