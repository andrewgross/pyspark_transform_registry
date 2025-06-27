FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates openjdk-17-jdk git && apt clean

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
