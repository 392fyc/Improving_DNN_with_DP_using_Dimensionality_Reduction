#!/bin/bash

# 替换下面为当前运行的容器名称或ID
CURRENT_CONTAINER="64728269a0c7b939d6a47355a8f476add76af57023624ce002d90574c13e72ce"

# 目标文件路径
HOST_PATH="D:/Research/Dimensionality_reduction_applied_Deep-Learning_under_Differential_Privacy/results/diabetic_jtree_results.csv"

# 源文件路径
CONTAINER_PATH="diabetic_jtree:/app/results.csv"


# 检测容器运行
while [ "$(docker container inspect -f '{{.State.Status}}' $CURRENT_CONTAINER)" == "running" ]; do
  echo "Waiting for the current container ($CURRENT_CONTAINER) to finish..."
  sleep 60 # 每60秒检查一次
done

echo "Current container ($CURRENT_CONTAINER) has stopped."
# 从容器中复制文件到宿主机
echo "Copying results from $CONTAINER_PATH to $HOST_PATH..."
docker cp $CONTAINER_PATH $HOST_PATH

# 容器停止后，在Windows环境下执行关机
cmd.exe /c shutdown /s /t 30