import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_anthropic import ChatAnthropic
from langchain.schema import Document

# 設定
CHROMA_HOST = "chroma"  # ChromaDBのホスト名
CHROMA_PORT = 8000  # ChromaDBのポート
COLLECTION_NAME = "code_chunks"  # コレクション名
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # 環境変数からAPIキーを取得

# ChromaDBクライアントの初期化
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# エンベディング関数の初期化
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# コレクションの取得
try:
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    print(f"コレクション '{COLLECTION_NAME}' を取得しました")
except Exception as e:
    print(f"コレクションの取得中にエラーが発生しました: {e}")
    print("まず /index エンドポイントを呼び出してコードベースのインデックスを作成してください")
    collection = None

# LLMの初期化
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    anthropic_api_key=ANTHROPIC_API_KEY
)

def query_code(question, k=5):
    """コードベースに対して質問を行い、回答と参照ソースを返す"""
    if collection is None:
        return {
            "result": "エラー: コレクションが初期化されていません。まず /index エンドポイントを呼び出してください。",
            "source_documents": []
        }
    
    # 質問をベクトル化して類似ドキュメントを検索
    results = collection.query(
        query_texts=[question],
        n_results=k
    )
    
    # 検索結果からドキュメントを作成
    source_documents = []
    for i in range(len(results["documents"][0])):
        doc = Document(
            page_content=results["documents"][0][i],
            metadata={
                "source": results["metadatas"][0][i]["source"],
                "file_path": results["metadatas"][0][i]["file_path"]
            }
        )
        source_documents.append(doc)
    
    # プロンプトの作成
    prompt = f"""
あなたはコードベースに関する質問に答えるアシスタントです。
以下のコードスニペットを参照して、質問に答えてください。

質問: {question}

参照コード:
"""
    
    for i, doc in enumerate(source_documents):
        prompt += f"\n--- スニペット {i+1} (ファイル: {doc.metadata['source']}) ---\n"
        prompt += doc.page_content + "\n"
    
    prompt += "\n上記のコードスニペットに基づいて、質問に対する回答を日本語で提供してください。"
    
    # LLMに質問を送信
    response = llm.invoke(prompt)
    
    # 結果を表示
    print("\n質問:")
    print(question)
    print("\n回答:")
    print(response.content)
    print("\n参照ソース:")
    for i, doc in enumerate(source_documents):
        print(f"\nソース {i+1}:")
        print(f"ファイル: {doc.metadata.get('source', 'Unknown')}")
        print(f"内容: {doc.page_content[:200]}...")
    
    return {
        "result": response.content,
        "source_documents": source_documents
    }

if __name__ == "__main__":
    # ユーザーからの質問を受け付ける
    while True:
        question = input("\nコードベースについての質問を入力してください（終了するには 'exit' と入力）: ")
        if question.lower() == "exit":
            break
        query_code(question) 