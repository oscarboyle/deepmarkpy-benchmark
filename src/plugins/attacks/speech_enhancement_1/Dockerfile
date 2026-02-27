FROM ml-services-base:latest

# Set ARG for build-time use defaults
ARG APP_PORT=10005
ENV APP_PORT=${SPEECH_ENHANCEMENT_PORT1}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/speech_enhancement_1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/attacks/speech_enhancement_1 .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT