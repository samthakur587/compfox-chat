# Use the official Python 3.11 image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY  requirements.txt .

RUN apt-get update && apt-get install -y tzdata


RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the test.py file to the container
COPY nya_file.py .



# Set the entrypoint command to run your FastAPI app
CMD ["uvicorn", "nya_file:app", "--host", "0.0.0.0"]
