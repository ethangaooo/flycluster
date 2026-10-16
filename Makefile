# FlyCluster 常用命令。所有目标都可用 `make <target>` 调用。
PY  ?= .venv/bin/python
PIP ?= .venv/bin/pip
NPM ?= npm

.PHONY: help setup smoke local-brain local-full experiment experiment-quick connectome-probe connectome-fetch admin-dev admin-build test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## 创建虚拟环境并安装依赖
	python3 -m venv .venv && $(PIP) install -q -r requirements-dev.txt

smoke: ## 端到端冒烟测试（不依赖网络与 Docker）
	PYTHONPATH=. $(PY) tools/smoke_test.py

local-brain: ## 单机模式启动中枢调度 + 开放 API（脑节点在进程内）
	PYTHONPATH=. FLYCLUSTER_API_KEYS=local:dev-local-key:600 \
	  $(PY) -m uvicorn services.orchestrator.app:app --host 127.0.0.1 --port 8800

docker-up: ## Docker 多容器集群（记忆中枢 + 环境 + 4 脑节点 + 调度）
	docker compose up --build

experiment-quick: ## 主对照快速跑通（结论不可用于决策）
	PYTHONPATH=. $(PY) experiments/run.py main --seeds 1 --quick

experiment: ## 主对照正式跑（默认 3 种子筛查）
	PYTHONPATH=. $(PY) experiments/run.py main --seeds 3

experiment-all: ## 主对照 + 实验 A/B/C
	PYTHONPATH=. $(PY) experiments/run.py all --seeds 3

connectome-probe: ## 探测官方 neuPrint 数据集可用性（需 NEUPRINT_TOKEN）
	PYTHONPATH=. $(PY) tools/fetch_connectome.py probe --dataset male-cns:v1.0

connectome-fetch: ## 抓取真实连接组子图
	PYTHONPATH=. $(PY) tools/fetch_connectome.py online --dataset male-cns:v1.0 \
	  --out data/male-cns-gateway.npz

admin-dev: ## 管理端开发服务器
	cd admin-ui && $(NPM) run dev

admin-build: ## 管理端构建到 admin-ui/dist
	cd admin-ui && $(NPM) run build

test: smoke ## 单元测试（当前以 21 项不变量的冒烟测试为唯一测试入口）

clean: ## 清理运行产物（不动源码与连接组数据）
	rm -rf artifacts/*.json artifacts/*.md data/local/*.jsonl data/*.sqlite3 data/dbg*
