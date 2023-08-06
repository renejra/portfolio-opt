#!/bin/bash
az container create --resource-group ${NAME_OF_RESOURCE_GROUP} --name ${APP_NAME} -f deployment.yml