# 招聘岗位RAG问答系统
> 基于本地向量检索实现的招聘岗位语义检索与RAG问答原型，支持**完全离线语义搜索**，可选择性对接大模型实现检索增强问答。

## ✨项目特性
1. **模块化Embedding工厂**：独立`embeddings.py`统一管理向量化实例，使用`lru_cache`实现模型单例加载，避免多脚本向量空间不一致问题，一处修改全局生效。
2. **离线语义检索**：不依赖大模型API，本地text2vec‑base‑chinese做Embedding，Chroma向量数据库完成相似度召回，返回匹配分数。
3. **RAG增强问答**：预留大模型接口，对接通义千问API，基于检索结果生成自然语言回答。
4. **配置解耦**：模型路径、参数、API密钥全部通过`.env`环境变量配置，不硬编码到代码。
5. 完善日志、异常捕获，项目提供requirements.txt，环境可快速复现。

## 📁项目目录结构
```text

rag_recruitment_qa/
├── data/                # 岗位 csv 数据源
├── models/              # 本地 text2vec 模型存放目录
├── chroma_db/           # Chroma 向量数据库（git 忽略，运行生成）
├── embeddings.py        # Embedding 单例工厂模块
├── build_vector_db.py   # 读取 csv，构建向量库
├── test_retrieve.py     # 【纯离线检索】交互式语义搜索，无需大模型
├── qa_cli.py            # 【RAG 问答】需要配置大模型 API‑key
├── .env                 # 本地环境配置
├── .env.example         # 环境变量模板
├── .gitignore
└── requirements.txt

```

## 🛠环境安装
```bash
# 创建虚拟环境（可选）
python -m venv .venv
# windows激活虚拟环境
.venv\Scripts\activate

# 安装全部依赖
pip install -r requirements.txt
```

## 📥模型准备

1. 前往 HuggingFace 下载` shibing624/text2vec-base-chinese `模型文件。
2. 将下载好的模型文件夹重命名为 `text2vec-base-chinese`。
3. 将其放置到项目根目录下的 `./models/ `文件夹中。

## ⚙环境变量配置

复制`.env.example`另存为`.env`

```
# embedding模型本地路径
EMBEDDING_MODEL=./models/text2vec-base-chinese

# Chroma向量数据库存储路径
CHROMA_PERSIST_DIR=./chroma_db

# 检索返回的文本块数量
TOP_K=5

# 调用RAG问答qa_cli时填写阿里云百炼密钥，离线检索不需要
DASHSCOPE_API_KEY=sk-xxx
```

## 🚀运行步骤

### 1、构建向量库

> 
> ⚠修改 embedding 参数后，需要删除 chroma_db 文件夹，重新执行构建

```
python build_vector_db.py
```

### 2、运行【纯离线语义检索】（推荐，无需任何 API 密钥）

```
python test_retrieve.py
```

- 输入查询语句做岗位语义搜索
- 输入`q`退出程序

### 3、运行【RAG 大模型问答】（需要配置 DASHSCOPE_API_KEY）

```
python qa_cli.py
```

## 💡说明

1. `test_retrieve.py`：纯本地向量召回，返回原始岗位元数据与相似度得分；Chroma 采用欧氏距离，**分数越小代表相似度越高**。
2. `qa_cli.py`：RAG 增强问答，在检索结果基础上调用大模型做整理生成，必须配置阿里云百炼 API 密钥，无密钥会返回 401 鉴权错误。
3. **禁止提交**：`models/`、`chroma_db/`、`.env`，已配置在 gitignore。

## 🧰技术栈

Python、LangChain、Chroma、sentence‑transformers (text2vec‑base‑chinese)、HuggingFace Embeddings、RAG、向量数据库、Pandas
