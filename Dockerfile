# Use official Python image as base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI's default port
EXPOSE 8002

# Start FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]