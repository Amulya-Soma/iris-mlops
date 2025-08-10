#!/bin/bash
set -e

IMAGE_NAME="${DOCKER_USERNAME:-amulyasoma}/iris-api:latest"
PORT=8005

echo "Pulling latest image from Docker Hub..."
docker pull "$IMAGE_NAME"

echo "Stopping any running container on port $PORT..."
CONTAINER_ID=$(docker ps -q --filter "publish=$PORT")
if [ -n "$CONTAINER_ID" ]; then
    docker stop "$CONTAINER_ID" && docker rm "$CONTAINER_ID"
fi

echo "Starting new container..."
docker run -d -p $PORT:8005 "$IMAGE_NAME"

echo "Deployment complete. App running at http://127.0.0.1:$PORT/docs"
