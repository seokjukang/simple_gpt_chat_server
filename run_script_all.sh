#!/bin/bash

: ${SERVICE_IP_KOALPACA="0.0.0.0"}
: ${SERVICE_PORT_KOALPACA=15000}

if [ "$1" == "start" ]; then
    echo "Starting model serving and chat service.."

    # nohup uvicorn _01_koalpaca_serving_fastapi:app --host $SERVICE_IP_KOALPACA --port $SERVICE_PORT_KOALPACA &
    # nohup uvicorn _02_bert_serving_fastapi:app --host $SERVICE_IP_BERT --port $SERVICE_PORT_BERT &
    nohup uvicorn chatserver_fastapi:app --host $SERVICE_IP_CHAT --port $SERVICE_PORT_CHAT &

elif [ "$1" == "stop" ]; then
    echo "Stopping model serving and chat service.."
    kill -9 `ps -ef|grep chatserver|awk '{print $2}'`

elif [ "$1" == "restart" ]; then
    echo "Restarting model serving and chat service.."
    kill -9 `ps -ef|grep chatserver|awk '{print $2}'`
    nohup uvicorn chatserver_fastapi:app --host $SERVICE_IP_CHAT --port $SERVICE_PORT_CHAT &
else
  echo "usage: ./start.sh start/stop/restart"
fi
