#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 561903439839.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 561903439839.dkr.ecr.ap-south-1.amazonaws.com/yt_chrome_plugin:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=yash-app)" ]; then
    echo "Stopping existing container..."
    docker stop yash-app
fi

if [ "$(docker ps -aq -f name=yash-app)" ]; then
    echo "Removing existing container..."
    docker rm yash-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name yash-app 561903439839.dkr.ecr.ap-south-1.amazonaws.com/yt_chrome_plugin:latest

echo "Container started successfully."