# 招聘岗位RAG问答系统
> 基于本地向量检索实现的招聘岗位语义检索与RAG问答原型，支持**完全离线语义搜索**，可选择性对接大模型实现检索增强问答。
> 
> 混合检索策略：BM25关键词检索 + 向量语义检索，RRF融合排序；自带离线召回评测脚本，**无需大模型API即可评估检索效果**。

## ✨项目特性
1. **模块化Embedding工厂**：独立`embeddings.py`统一管理向量化实例，使用`lru_cache`实现模型单例加载，避免多脚本向量空间不一致问题，一处修改全局生效。
2. **离线语义检索**：不依赖大模型API，本地`text2vec-base-chinese`做Embedding，Chroma向量数据库完成相似度召回，返回匹配分数。
3. **RAG增强问答**：预留大模型接口，对接通义千问API，基于检索结果生成自然语言回答。
4. **配置解耦**：模型路径、参数、API密钥全部通过`.env`环境变量配置，不硬编码到代码。
5. **完善日志、异常捕获**，项目提供`requirements.txt`，环境可快速复现。
6. **自动化离线评测**：内置Context Recall召回率计算脚本，Docker环境可一键执行评测，不消耗LLM Token，评测结果自动落地本地文件。
7. **Docker一键容器化部署**：目录挂载持久化，向量库、评测报告直接输出至本机，容器销毁数据不丢失。

## 📁项目目录结构
```text
rag_recruitment_qa/
├── data/                # 岗位 csv 数据源
├── models/              # 本地 text2vec 模型存放目录
├── chroma_db/           # Chroma 向量数据库（git忽略，运行生成）
├── core/
│   ├── embeddings.py    # Embedding 单例工厂模块
│   ├── config.py        # 环境配置读取
│   └── retriever.py     # BM25+向量混合检索逻辑
├── scripts/
│   ├── build_vector_db.py # 读取 csv，构建向量库
│   └── test_retrieve.py # 【纯离线检索】交互式语义搜索，无需大模型
├── eval/
│   ├── qa_dataset.json  # 评测问答样本集
│   └── run_eval.py      # 离线召回率评测脚本，输出report.txt
├── qa_cli.py            # 【RAG 问答】需要配置大模型 API-key
├── .env                 # 本地环境配置（不上传）
├── .env.example         # 环境变量模板
├── .gitignore
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```
## 🧰 环境安装（本地 Python 运行）
```bash
# 创建虚拟环境（可选）
python -m venv .venv
# Windows激活虚拟环境
.venv\Scripts\activate

# 安装全部依赖
pip install -r requirements.txt
```

## 📦模型准备
1. 前往 HuggingFace 下载 shibing624/text2vec-base-chinese 模型文件。
2. 将下载好的模型文件夹重命名为 text2vec-base-chinese。
3. 将其放置到项目根目录下的 ./models/ 文件夹中。

## ⚙️环境变量配置
复制 .env.example 另存为 .env
```env
# embedding模型本地路径
EMBEDDING_MODEL=./models/text2vec-base-chinese

# Chroma向量数据库存储路径
CHROMA_PERSIST_DIR=./chroma_db

# 检索返回的文本块数量
TOP_K=5

# 调用RAG问答qa_cli.py时填写阿里云百炼密钥，离线检索不需要
DASHSCOPE_API_KEY=sk-xxx
```
## 🚀运行步骤
1. 构建向量库
>
>⚠️ 修改 embedding 参数 / 更换知识库数据后，需要删除 chroma_db 文件夹，重新执行构建
```
python scripts/build_vector_db.py
```
## 📊 检索离线评测（无需 LLM API）
用于自动化验证检索召回效果，Docker CI 环境也可以执行，不消耗 token
评测文件路径：eval/qa_dataset.json
> 
>⚠️ 注意：评测样本的答案关键词必须存在于知识库 csv 数据内；更换自己的岗位数据包时，需要同步修改qa_dataset.json，否则召回率会归零。
```
# 执行离线检索评测
python eval/run_eval.py
```
执行完成后，评测指标结果自动写入 eval/report.txt，直接打开查看。

2. 运行【纯离线语义检索】(推荐，无需任何 API 密钥)
```
python scripts/test_retrieve.py
```
- 输入查询语句做岗位语义搜索
- 输入q退出程序

3. 运行【RAG 大模型问答】（需要配置 DASHSCOPE_API_KEY）
```
python qa_cli.py
```

## 🐳 Docker 一键部署
>项目提供docker-compose.yml，可容器化一键部署，环境完全隔离。
> 
>挂载策略：data、models、chroma_db、eval 目录全部挂载到本地，容器运行产生的向量库、评测报告直接保存在本机，容器删除不会丢失结果。
```
# 构建镜像（仅修改Dockerfile/requirements.txt才需要加--no-cache；单纯更换数据无需重新build）
docker compose build --no-cache

# 前台启动（推荐调试，实时看日志）
docker compose up

# 后台守护运行
docker compose up -d

# 查看运行日志
docker compose logs -f rag-recruit
```

## 🎬 演示一键复现命令：
```
# 清理旧容器与旧向量库
docker compose down
# Windows PowerShell 删除本地旧向量库
Remove-Item -Recurse -Force ./chroma_db

# 启动容器，自动构建向量库并执行评测
docker compose up
```

> ✅ 容器跑完评测后，直接打开本地 eval/report.txt 查看召回结果，无需 docker cp 拷贝文件
> 
## 📝 更换数据集快速流程
> 仅替换data内岗位 csv 数据、代码 / 依赖无改动时：
> 
1. 删除旧向量库：Remove-Item -Recurse -Force ./chroma_db
2. 本地测试：python scripts/build_vector_db.py → python eval/run_eval.py
3. 验证无误后，直接docker compose up，容器会自动读取新数据重建向量库并评测

## 💡说明
1. test_retrieve.py：纯本地向量召回，返回原始岗位元数据与相似度得分；Chroma 采用欧氏距离，分数越小代表相似度越高。
2. qa_cli.py：RAG 增强问答，在检索结果基础上调用大模型做整理生成，必须配置阿里云百炼 API 密钥，无密钥会返回 401 鉴权错误。
禁止提交：models/、chroma_db/、.env，已配置在 .gitignore。

## 🛠 技术栈
Python、LangChain、Chroma、sentence-transformers（text2vec-base-chinese）、HuggingFace Embeddings、RAG、向量数据库、Pandas、BM25 混合检索