#!/bin/bash

# 替换下面为当前运行的容器名称或ID
CURRENT_CONTAINER="8ba4c9d9957232d20ffb06b819d9377dbcf1003244d33f74936628ec70cd69f4"

# 替换下面的COMMAND_TO_RUN_NEXT_CONTAINER为下一个容器的完整docker run命令
NEXT_CONTAINER_COMMAND="docker run --gpus all --name mnist_fashion_1_pca_500 mnist_fashion_pca_500_dp"

# 目标文件路径
HOST_PATH="D:/Research/Dimensionality_reduction_applied_Deep-Learning_under_Differential_Privacy/results/pca_5_results.csv"

# 源文件路径
CONTAINER_PATH="mnist_fashion_1_pca_5:/app/training_results.csv"



# 检测容器运行
while [ "$(docker container inspect -f '{{.State.Status}}' $CURRENT_CONTAINER)" == "running" ]; do
  echo "Waiting for the current container ($CURRENT_CONTAINER) to finish..."
  sleep 480 # 每480秒检查一次
done

echo "Current container ($CURRENT_CONTAINER) has stopped."
# 从容器中复制文件到宿主机
echo "Copying results from $CONTAINER_PATH to $HOST_PATH..."
docker cp $CONTAINER_PATH $HOST_PATH

echo "Starting the next container..."
# 运行下一个容器
eval $NEXT_CONTAINER_COMMAND
