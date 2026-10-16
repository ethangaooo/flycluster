# 单一镜像，多角色：compose 里通过 command 指定启动哪个服务。
# 这样脑节点 / 编排器 / 记忆中枢 / 环境共享同一份代码与依赖，不会出现版本漂移。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# 默认走官方 PyPI。国内网络慢或需要走内网源时，不改文件，构建时覆盖即可：
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 仓库里**不硬编码**任何第三方镜像地址，避免把环境假设固化进代码。
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements.txt

COPY flycore ./flycore
COPY services ./services
COPY experiments ./experiments
COPY tools ./tools

# 数据卷挂载点：记忆数据库、本地经验、运行产物都落在这里
RUN mkdir -p /app/data /app/artifacts
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "services.orchestrator.app:app", "--host", "0.0.0.0", "--port", "8000"]
