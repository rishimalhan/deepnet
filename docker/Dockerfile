# Use an official Python image with ARM support as the base
FROM python:3.11-slim

# Set environment variables for better performance
ENV OMP_NUM_THREADS=8
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app
COPY . .
COPY ./docker/packages-python3.txt .
COPY ./docker/packages-apt.txt .

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    libopenblas-dev \
    libomp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir -r packages-python3.txt
RUN apt-get update && xargs -a packages-apt.txt apt-get install -y

RUN echo "alias python=python3" >> /root/.bashrc
RUN chmod 755 /app/docker/entrypoint.sh

EXPOSE 8080

RUN echo 'export PYTHONPATH="${PYTHONPATH}:/app/src/"' >> /root/.bashrc

ENTRYPOINT ["bash",  "/app/docker/entrypoint.sh" ]