FROM ubuntu:20.04

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY ./docker/packages-python3.txt .
COPY ./docker/packages-apt.txt .


# Install any additional system packages needed
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && xargs -a packages-apt.txt apt-get install -y

# Install any additional Python packages needed
RUN apt-get update && apt-get install python3 -y

RUN echo "alias python=python3" >> /root/.bashrc

RUN apt-get update && apt-get install -y python3-pip

RUN pip install --no-cache-dir -r packages-python3.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Copy the rest of the application code into the container
COPY . .

RUN git clone https://github.com/Plachtaa/VALL-E-X.git
WORKDIR /app/VALL-E-X
RUN pip install -r requirements.txt
WORKDIR /app/
RUN rm -rf VALL-E-X

RUN pip install --upgrade openai alive-progress scikit-learn tqdm openai-whisper ffmpeg-python setuptools-rust

RUN echo "alias python=python3" >> /root/.bashrc
RUN chmod 755 /app/docker/entrypoint.sh

EXPOSE 8080
RUN apt-get install libportaudio2

ENTRYPOINT ["bash",  "/app/docker/entrypoint.sh" ]