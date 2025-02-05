FROM python:3.9-slim

# Prevent Python from writing pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app

# Expose the port (Render sets the PORT env variable; default to 5000 if not set)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"] 