# Use Python 3.9-slim as the base image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the local directory contents (Flask app files) to the container's work directory
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download and install the spaCy English large model
RUN python -m spacy download en_core_web_lg

# Command to run the application
CMD ["python", "app.py"]
