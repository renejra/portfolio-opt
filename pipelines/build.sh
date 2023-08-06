#!/bin/bash
docker build -t ${APP_NAME}:latest --platform=linux/amd64 .
docker tag ${APP_NAME}:latest ${NAME_OF_REGISTRY}.azurecr.io/${APP_NAME}:latest
docker push ${NAME_OF_REGISTRY}.azurecr.io/${APP_NAME}:latest