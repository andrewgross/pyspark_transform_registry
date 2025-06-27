FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates openjdk-17-jdk && apt clean

# Setup User
RUN useradd -m -s /bin/bash ubuntu
USER ubuntu
WORKDIR /home/ubuntu

# Install UV
ADD https://astral.sh/uv/install.sh /home/ubuntu/uv-installer.sh
RUN sh /home/ubuntu/uv-installer.sh && rm /home/ubuntu/uv-installer.sh
ENV PATH="/home/ubuntu/.local/bin/:$PATH"

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
