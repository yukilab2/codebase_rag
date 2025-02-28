import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import pytesseract
from PIL import Image
import io
import pdfplumber  # PyMuPDFの代わりにpdfplumberを使用
import re

# 設定
SOURCE_CODE_DIR = "/code_repo"  # コンテナ内のソースコードディレクトリ
DOCS_DIR = os.path.join(SOURCE_CODE_DIR, "docs")  # ドキュメントディレクトリ
CHROMA_PERSIST_DIR = "/app/chroma_db"  # ChromaDBの保存先
CHROMA_HOST = "chroma"  # ChromaDBのホスト名
CHROMA_PORT = 8000  # ChromaDBのポート
COLLECTION_NAME = "code_chunks"  # コレクション名
CHUNK_SIZE = 1000  # テキストチャンクのサイズ
CHUNK_OVERLAP = 200  # チャンク間のオーバーラップ
EXTENSIONS = [".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php"]  # 対象とするファイル拡張子
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]  # 対象とする画像ファイル拡張子
PDF_EXTENSIONS = [".pdf"]  # 対象とするPDFファイル拡張子

# ディレクトリが存在しない場合は作成
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ChromaDBクライアントの初期化
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# エンベディング関数の初期化
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# コレクションの作成または取得
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"既存のコレクション '{COLLECTION_NAME}' を取得しました")
except:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    print(f"新しいコレクション '{COLLECTION_NAME}' を作成しました")

# テキスト分割器の初期化
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

def get_all_code_files():
    """対象となるすべてのコードファイルのパスを取得"""
    all_files = []
    for ext in EXTENSIONS:
        pattern = os.path.join(SOURCE_CODE_DIR, f"**/*{ext}")
        all_files.extend(glob.glob(pattern, recursive=True))
    return all_files

def get_all_doc_files():
    """docsディレクトリ内のすべての画像とPDFファイルのパスを取得"""
    if not os.path.exists(DOCS_DIR):
        print(f"ドキュメントディレクトリ {DOCS_DIR} が存在しません")
        return []
    
    all_files = []
    # 画像ファイルを取得
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(DOCS_DIR, f"**/*{ext}")
        all_files.extend(glob.glob(pattern, recursive=True))
    
    # PDFファイルを取得
    for ext in PDF_EXTENSIONS:
        pattern = os.path.join(DOCS_DIR, f"**/*{ext}")
        all_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"検索パターン: {DOCS_DIR}/**/*[{','.join(IMAGE_EXTENSIONS + PDF_EXTENSIONS)}]")
    return all_files

def process_file(file_path):
    """ファイルを読み込み、チャンクに分割してドキュメントを作成"""
    try:
        # ファイルの相対パスを取得（メタデータ用）
        rel_path = os.path.relpath(file_path, SOURCE_CODE_DIR)
        
        # ファイルを読み込む
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # 各ドキュメントにファイルパスのメタデータを追加
        for doc in documents:
            doc.metadata["source"] = rel_path
            doc.metadata["file_path"] = file_path
        
        # ドキュメントをチャンクに分割
        chunks = text_splitter.split_documents(documents)
        
        print(f"処理中: {rel_path} - {len(chunks)}チャンクに分割")
        return chunks
    except Exception as e:
        print(f"エラー: {file_path}の処理中に問題が発生しました: {e}")
        return []

def process_image(file_path):
    """画像ファイルからテキストを抽出してドキュメントを作成"""
    try:
        # ファイルの相対パスを取得（メタデータ用）
        rel_path = os.path.relpath(file_path, SOURCE_CODE_DIR)
        
        # 画像を読み込む
        image = Image.open(file_path)
        
        # 画像からテキストを抽出
        extracted_text = pytesseract.image_to_string(image, lang='jpn+eng')
        
        # 空のテキストの場合はスキップ
        if not extracted_text.strip():
            print(f"警告: {rel_path} からテキストを抽出できませんでした")
            return []
        
        # ドキュメントを作成
        from langchain.schema import Document
        doc = Document(
            page_content=extracted_text,
            metadata={
                "source": rel_path,
                "file_path": file_path,
                "type": "image"
            }
        )
        
        # ドキュメントをチャンクに分割
        chunks = text_splitter.split_documents([doc])
        
        print(f"処理中: {rel_path} - {len(chunks)}チャンクに分割")
        return chunks
    except Exception as e:
        print(f"エラー: {file_path}の処理中に問題が発生しました: {e}")
        return []

def process_pdf(file_path):
    """PDFファイルからテキストを抽出してドキュメントを作成"""
    try:
        # ファイルの相対パスを取得（メタデータ用）
        rel_path = os.path.relpath(file_path, SOURCE_CODE_DIR)
        
        # PDFを開く
        extracted_text = ""
        with pdfplumber.open(file_path) as pdf:
            # すべてのページからテキストを抽出
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n\n"
        
        # 空のテキストの場合はスキップ
        if not extracted_text.strip():
            print(f"警告: {rel_path} からテキストを抽出できませんでした")
            return []
        
        # ドキュメントを作成
        from langchain.schema import Document
        doc = Document(
            page_content=extracted_text,
            metadata={
                "source": rel_path,
                "file_path": file_path,
                "type": "pdf"
            }
        )
        
        # ドキュメントをチャンクに分割
        chunks = text_splitter.split_documents([doc])
        
        print(f"処理中: {rel_path} - {len(chunks)}チャンクに分割")
        return chunks
    except Exception as e:
        print(f"エラー: {file_path}の処理中に問題が発生しました: {e}")
        return []

def main():
    all_chunks = []
    
    # すべてのコードファイルを取得して処理
    code_files = get_all_code_files()
    print(f"{len(code_files)}個のコードファイルが見つかりました")
    
    for file_path in code_files:
        chunks = process_file(file_path)
        all_chunks.extend(chunks)
    
    # すべてのドキュメントファイル（画像とPDF）を取得して処理
    doc_files = get_all_doc_files()
    print(f"{len(doc_files)}個のドキュメントファイルが見つかりました")
    
    for file_path in doc_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in IMAGE_EXTENSIONS:
            chunks = process_image(file_path)
            all_chunks.extend(chunks)
        elif file_ext in PDF_EXTENSIONS:
            chunks = process_pdf(file_path)
            all_chunks.extend(chunks)
    
    print(f"合計{len(all_chunks)}チャンクを処理しました")
    
    # ChromaDBにデータを保存
    if all_chunks:
        # ChromaDBに追加するためのデータを準備
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            metadatas.append({
                "source": chunk.metadata.get("source", "Unknown"),
                "file_path": chunk.metadata.get("file_path", "Unknown"),
                "type": chunk.metadata.get("type", "code")
            })
        
        # ChromaDBにデータを追加
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"ChromaDBに{len(all_chunks)}チャンクを保存しました")
    else:
        print("保存するチャンクがありません")

if __name__ == "__main__":
    main() 