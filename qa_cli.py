#qa_cli.py
import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from embeddings import get_embeddings

# 兼容新旧版本
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_qa_chain():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    # 必须和构建向量库完全一样的Embedding类！！
    embeddings = get_embeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    # embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="recruit_job"
    )

    llm = ChatOpenAI(
        model=os.getenv("QWEN_LLM_MODEL", "qwen-plus"),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1,
        request_timeout=60
    )

    prompt_template = """你是招聘领域的智能问答助手。请根据以下岗位信息回答用户的问题。
如果提供的信息中没有相关内容，请诚实回答 "没有找到相关岗位"。

参考信息：
{context}

用户问题：{question}
你的回答："""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return qa_chain

def main():
    logger.info("启动招聘问答助手...")
    try:
        qa = build_qa_chain()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    print("\n" + "=" * 50)
    print("🎯 岗位智能问答助手已启动！")
    print("（输入 'q' 退出）")
    print("=" * 50)

    while True:
        query = input("\n请输入你的问题: ").strip()
        if query.lower() in ('q', 'quit', 'exit'):
            print("再见！")
            break
        if not query:
            continue

        print("\n正在思考...")
        try:
            response = qa.invoke({"query": query})
            answer = response["result"]
            print(f"\n💡 回答: {answer}")
        except Exception as e:
            logger.error(f"问答出错: {e}")
            print("\n⚠️ 暂时无法回答，请稍后再试。")

if __name__ == "__main__":
    main()
