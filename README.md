# 掌柜智库 Shopkeeper Brain

> 基于 **LangGraph** + 多模态大模型的智能知识库管理与查询系统。
> 把一堆 PDF / Markdown / 图片说明书丢进去，自动切分、向量化、建知识图谱，然后用混合检索 + 流式问答把答案取回来。

一个面向「店铺经营知识」场景的 RAG 系统：导入侧负责把非结构化文档变成可检索的知识资产，查询侧负责把口语化的问题变成有依据的答案。

---

## 功能特性

### 文档导入（`:8000`）
- 支持 **PDF / Markdown / 图片** 多格式，PDF 走 MinerU 做版面分析，保留表格、公式、插图
- Markdown 里的图片自动上传 **MinIO** 并回写为可访问 URL
- 按标题层级做**语义切分**，产出带元数据的 chunk
- LLM 自动识别文档所属**商品名**，作为后续检索的过滤维度
- 自动抽取实体与关系，写入 **Neo4j** 构建知识图谱
- 密集 + 稀疏**双向量**写入 **Milvus**，支持混合检索

### 智能查询（`:8001`）
- **商品名确认**：口语化提问先对齐到知识库里的标准商品名，避免"检索跑偏"；不明确时反问澄清
- **四路混合检索**并行执行：
  | 通路 | 作用 |
  |---|---|
  | 向量检索 | BGE-M3 稀疏+稠密混合召回 |
  | HyDE | 先让 LLM 生成"假设性文档"再检索，弥补问题与文档的表述差异 |
  | 知识图谱 | 实体抽取 → 对齐 → Neo4j 一跳关系 → 反查 chunk |
  | 网络搜索 | 知识库里没有时走 MCP 联网兜底 |
- **RRF 融合**（Reciprocal Rank Fusion）+ **BGE 重排序**，多路结果统一排序
- **SSE 流式输出**：逐 token 推送 `delta`，实时回传节点进度
- **多轮会话**：历史对话存 MongoDB，支持代词指代消解与 `item_names` 回填

---

## 技术架构

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI  API 层                           │
│         导入服务 :8000 (/import)   查询服务 :8001 (/chat.html) │
├──────────────────────────────────────────────────────────────┤
│                   LangGraph  工作流引擎                        │
│      ┌──────────────────────┬──────────────────────┐         │
│      │  导入流程（8 节点）    │   查询流程（9 节点）   │         │
│      └──────────────────────┴──────────────────────┘         │
├──────────────────────────────────────────────────────────────┤
│                          工具层                                │
│   LLM 客户端 │ BGE 向量化 │ Rerank │ Milvus │ Neo4j │ MinIO     │
│   SSE 推送   │ 任务追踪   │ Mongo 历史 │ 路径/配置              │
├──────────────────────────────────────────────────────────────┤
│                          存储层                                │
│   Milvus(向量)  Neo4j(图谱)  MongoDB(历史)  MinIO(对象存储)      │
└──────────────────────────────────────────────────────────────┘
```

### 技术栈

| 类别 | 选型 |
|---|---|
| 后端框架 | FastAPI + Uvicorn |
| 工作流引擎 | LangGraph（状态图 / 条件边 / 并行分支） |
| 向量数据库 | Milvus（稠密 + 稀疏混合检索，WeightedRanker 归一融合） |
| 图数据库 | Neo4j |
| 文档数据库 | MongoDB（会话历史） |
| 对象存储 | MinIO |
| 嵌入模型 | BGE-M3（dense + sparse） |
| 重排序模型 | BGE-Reranker-Large（FlagEmbedding） |
| 大模型 | DeepSeek（`langchain-deepseek`）+ DashScope 视觉模型 |
| PDF 解析 | MinerU |
| 前端 | 原生 HTML + JS（无构建步骤） |

---

## 目录结构

两个流程各自是一个**自包含单元**：自己的 `api / core / front / services / schema / nodes`，只共享 `utils/`。

```
shopkeeper-brain/
├─ knowledge/
│  ├─ processor/
│  │  ├─ import_process/            # 导入流程（:8000）
│  │  │  ├─ api/import_router.py    # FastAPI 应用与路由
│  │  │  ├─ core/{deps,paths}.py
│  │  │  ├─ front/import.html       # 上传页
│  │  │  ├─ nodes/                  # LangGraph 节点 + main_graph.py
│  │  │  ├─ prompt/                 # 提示词
│  │  │  ├─ services/               # ImportFileService / TaskService
│  │  │  ├─ shcema/                 # 请求/响应模型
│  │  │  ├─ base.py  config.py  exceptions.py  state.py
│  │  └─ query_process/             # 查询流程（:8001）
│  │     ├─ api/query_router.py     # FastAPI 应用与路由
│  │     ├─ core/{deps,paths}.py
│  │     ├─ front/chat.html         # 聊天页
│  │     ├─ nodes/                  # LangGraph 节点 + main_graph.py
│  │     ├─ prompts/                # 提示词
│  │     ├─ services/query_service.py
│  │     ├─ shcema/query_schema.py
│  │     ├─ base.py  config.py  exceptions.py  state.py
│  ├─ utils/                       # 共享工具
│  │  ├─ llm_client.py             # LLM 客户端
│  │  ├─ bge_m3_embedding*.py      # BGE-M3 向量化
│  │  ├─ bge_rerank_util.py        # 重排序
│  │  ├─ milvus_util.py            # Milvus 混合检索封装
│  │  ├─ neo4j_util.py  minio_util.py
│  │  ├─ mongo_history_util.py     # 会话历史读写
│  │  ├─ sse_util.py  task_util.py # SSE 队列 / 任务状态
│  │  └─ markdown_utils.py
│  ├─ .env                         # 配置（不要提交）
│  ├─ .env.example                 # 配置模板
│  └─ requirements.txt
├─ start_import_8000.bat           # 一键启动导入服务
├─ start_query_8001.bat            # 一键启动查询服务
└─ README.md
```

---

## 快速开始

### 0. 克隆仓库

```bash
git clone https://github.com/jinbinyou8-lab/shopkeeper-brain.git
cd shopkeeper-brain
```

### 1. 依赖服务

| 服务 | 默认地址 | 用途 |
|---|---|---|
| Milvus | `127.0.0.1:19530` | 向量库 |
| Neo4j | `127.0.0.1:7687` / `:7690` | 知识图谱 |
| MongoDB | `127.0.0.1:27017` | 会话历史 |
| MinIO | `127.0.0.1:9000` | 图片/对象存储 |

### 2. 安装依赖

```bash
cd knowledge
python -m venv .venv
.venv\Scripts\activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

> 需要 GPU 环境（BGE-M3 / Reranker / MinerU 都跑在 `cuda:0`）。

### 3. 配置

```bash
copy knowledge\.env.example knowledge\.env      # Windows
cp knowledge/.env.example knowledge/.env        # Linux/macOS
```

按实际情况填 `.env`，至少要有：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 查询/导入用的 LLM |
| `MONGO_URL` / `MONGO_DB_NAME` | 会话历史，不配则历史功能整体不可用 |
| `MILVUS_URL` | Milvus 地址 |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j 连接 |
| `MINIO_ENDPOINT` / `MINIO_BUCKET_NAME` | MinIO 地址与桶名 |
| `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` / `ENTITY_NAME_COLLECTION` | 三个集合名 |
| `BGE_RERANKER_LARGE` | 重排序模型本地路径 |

BGE-M3 的模型路径在 `utils/bge_m3_embedding*.py` 里，按需改成自己的目录。

### 4. 启动

**推荐**：直接双击这两个脚本（已自动处理工作目录、`PYTHONPATH` 和代理变量）

```
start_import_8000.bat     →  导入服务
start_query_8001.bat      →  查询服务
```

**或者手动**（注意必须在**项目根目录**下启动，且要让 `knowledge` 包可被导入）：

```bash
cd shopkeeper_brain

# Linux/macOS
PYTHONPATH=$(pwd) python -m uvicorn \
  knowledge.processor.query_process.api.query_router:create_app \
  --factory --host 0.0.0.0 --port 8001

# Windows PowerShell
$env:PYTHONPATH=(Get-Location).Path
python -m uvicorn knowledge.processor.query_process.api.query_router:create_app `
  --factory --host 0.0.0.0 --port 8001
```

> ⚠️ 首次启动需要约 60 秒（要加载 torch / FlagEmbedding / pymilvus 模型库），看到 `Application startup complete` 才算就绪。
>
> ⚠️ 如果本机开着系统代理，请先清掉 `HTTP_PROXY / HTTPS_PROXY`：gRPC 也会走代理，会导致 Milvus 连接报 502。

### 5. 打开前端

| 页面 | 地址 |
|---|---|
| 知识库问答 | http://127.0.0.1:8001/ |
| 文档导入 | http://127.0.0.1:8000/import |
| 接口文档 | http://127.0.0.1:8001/docs |

---

## 核心流程

### 导入流程（8 节点）

```
entry_node ──┬─ is_pdf_read_enabled ─> pdf_to_md_node ─┐
             ├─ is_md_read_enabled  ─> md_img_node <───┘
             └─ 其它                ─> END
                                        │
                                        v
                    md_img_node ─> document_split ─> item_name_node
                                        │
                                        v
              bge_embedding_chunks ─> import_milvus_node ─> kg_graph_node ─> END
```

### 查询流程（9 节点）

```
                      item_name_confirm
                            │
              ┌── 已有答案 ─┴─ 需要检索 ───────────────┐
              │              │                       │
              │              v                       │
              │         multi_search                 │
              │              │                       │
              │   ┌──────────┼──────────┬─────────┐  │
              │   v          v          v         v  │
              │ vector      hyde       kg      web搜索│
              │   └──────────┴──────────┴─────────┘  │
              │              │                       │
              │              v                       │
              └──────────>  rrf ─> rerank ─> answer_output ─> END
```

设计上有两点值得注意：
- **并行分支只返回自己修改的字段**（`return {"embedding_chunks": ...}`）。并行节点若返回整个 state，LangGraph 会抛 `InvalidUpdateError: At key 'session_id': Can receive only one value per step`。
- **SSE 三层解耦**：`task_util` 只管任务状态、`sse_util` 只管消息推送、`BaseNode` 是唯一同时知道两者的地方，负责协调进度上报。

---

## 接口一览

### 查询服务 `:8001`

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 提交查询；`is_stream=false` 同步返回答案，`true` 返回 `task_id` |
| GET | `/stream/{task_id}` | SSE 长连接：`ready` / `progress` / `delta` / `final` |
| GET | `/status/{task_id}` | 轮询任务状态与进度 |
| GET | `/history/{session_id}` | 拉取会话历史 |
| DELETE | `/history/{session_id}` | 清空会话历史 |
| GET | `/health` | 健康检查 |
| GET | `/chat.html` `GET /` | 聊天前端 |

### 导入服务 `:8000`

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 上传文档，返回 `task_id`，后台跑导入图 |
| GET | `/status/{task_id}` | 查询导入进度 |
| GET | `/import` | 上传前端 |

---

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 启动很慢 | 正常，首次导入要拉起 torch 等重型库，约 60s |
| Milvus 报 `illegal connection params or server unavailable` 且日志有 `HTTP proxy returned response code 502` | 系统代理劫持了 gRPC，清掉 `HTTP_PROXY/HTTPS_PROXY` 再启动 |
| 问答返回「抱歉，我无法识别您询问的具体产品名称」 | 这是降级话术：LLM 没抽到商品名，或 Milvus 连不上导致商品名对齐失败。先确认向量库在跑、集合已导入数据 |
| 进度条一直显示英文节点名 | `task_util.py` 的 `_NODE_NAME_TO_CN` 键名要与 `BaseNode.name` 一致 |
| `/history` 返回 500 | MongoDB 未配置；`MONGO_URL` / `MONGO_DB_NAME` 缺失 |
| 历史里 `item_names` 一直是空 | `item_name_confirm_node` 的 `update_message_item_names` 回填没生效 |

---

## 说明

本项目为个人学习 / 实战项目，需求与课程框架来自开源教程仓库 [2019hzk/shopkeeper_brain](https://github.com/2019hzk/shopkeeper_brain)；
在原有基础上做了一轮结构重构（按流程拆成自包含的 `import_process` / `query_process`）、补齐了查询侧的 API / Service / Schema / 前端链路，并修复了并行分支、节点初始化、工具模块命名等一系列运行期问题。

仅用于学习交流。
