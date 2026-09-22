# 使用 Python 3.10 作为基础镜像
FROM m.daocloud.io/docker.io/library/python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量：防止生成 pyc 缓存文件，强制标准输出无缓冲，并设置项目根目录到模块搜索路径
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 复制依赖文件并安装（使用清华源加速下载）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 将当前项目根目录下所有文件复制到容器中
COPY . .

# 默认启动命令（容器启动时自动运行）
CMD ["python", "eval/run_eval.py"]