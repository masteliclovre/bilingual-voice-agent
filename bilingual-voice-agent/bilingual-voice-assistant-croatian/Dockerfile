# Use an official Python 3.11 runtime as a parent image.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg, which is a crucial system dependency for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Copy the model download script.
COPY download-model.py .

# Run the download script to fetch the fine-tuned model.
RUN python download-model.py

# Copy the rest of the application code.
COPY app.py .

# Expose the port the app runs on.
EXPOSE 8000

# Define the command to run the application using the Gunicorn production server.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "app:app"]

