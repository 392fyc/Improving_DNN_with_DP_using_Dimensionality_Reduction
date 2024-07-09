# 使用与TensorFlow 2.5.0兼容的官方GPU镜像作为基础镜像
FROM tensorflow/tensorflow:2.5.0-gpu

# 将当前目录下的所有文件复制到容器中的/app目录下
COPY . /app

# 设置工作目录为/app
WORKDIR /app

# 安装requirements.txt指定的所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 指定容器启动时运行的命令
CMD ["python", "./mnist_fashion_APCA.py"]