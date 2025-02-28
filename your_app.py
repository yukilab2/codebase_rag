from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
import subprocess
import base64
from typing import Optional, List, Dict, Any
import time

# 画像処理用のライブラリをインポート
try:
    from PIL import Image
    import pytesseract
    import io
    import numpy as np
    import cv2
    HAS_IMAGE_PROCESSING = True
except ImportError:
    HAS_IMAGE_PROCESSING = False

app = FastAPI()

# HTMLテンプレートディレクトリの設定
templates = Jinja2Templates(directory="templates")

# インデックス作成の状態を管理するグローバル変数
indexing_status = {
    "is_running": False,
    "start_time": None,
    "end_time": None,
    "status": "idle",  # idle, running, completed, error
    "message": "",
    "error": None
}

class QueryRequest(BaseModel):
    question: str

class IndexResponse(BaseModel):
    status: str

class IndexStatusResponse(BaseModel):
    is_running: bool
    status: str
    message: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ImageQueryRequest(BaseModel):
    image_data: str  # Base64エンコードされた画像データ
    question: Optional[str] = None  # 画像に関する質問（オプション）

# インデックス作成のバックグラウンドタスク
def run_indexer():
    global indexing_status
    
    try:
        # 状態を更新
        indexing_status["is_running"] = True
        indexing_status["start_time"] = time.time()
        indexing_status["status"] = "running"
        indexing_status["message"] = "インデックス作成を実行中..."
        indexing_status["error"] = None
        
        # インデックス作成を実行
        result = subprocess.run(["python", "code_indexer.py"], check=True, capture_output=True, text=True)
        
        # 成功した場合の状態更新
        indexing_status["status"] = "completed"
        indexing_status["message"] = "インデックス作成が完了しました"
        indexing_status["end_time"] = time.time()
    except subprocess.CalledProcessError as e:
        # エラーが発生した場合の状態更新
        indexing_status["status"] = "error"
        indexing_status["message"] = f"インデックス作成中にエラーが発生しました"
        indexing_status["error"] = str(e) + "\n" + e.stdout + "\n" + e.stderr
        indexing_status["end_time"] = time.time()
    finally:
        # 実行状態を更新
        indexing_status["is_running"] = False

# コードベースのインデックスを作成するエンドポイント
@app.post("/index", response_model=IndexResponse)
async def index_code(background_tasks: BackgroundTasks):
    global indexing_status
    
    # 既に実行中の場合はエラーを返す
    if indexing_status["is_running"]:
        return {"status": "already_running", "message": "インデックス作成は既に実行中です"}
    
    # バックグラウンドでインデックス作成を実行
    background_tasks.add_task(run_indexer)
    return {"status": "processing", "message": "コードベースのインデックス作成を開始しました。これには数分かかる場合があります。"}

# インデックス作成の状態を確認するエンドポイント
@app.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    global indexing_status
    
    response = {
        "is_running": indexing_status["is_running"],
        "status": indexing_status["status"],
        "message": indexing_status["message"],
        "start_time": indexing_status["start_time"],
        "end_time": indexing_status["end_time"],
    }
    
    # 所要時間を計算
    if indexing_status["start_time"] is not None:
        if indexing_status["end_time"] is not None:
            response["duration"] = indexing_status["end_time"] - indexing_status["start_time"]
        elif indexing_status["is_running"]:
            response["duration"] = time.time() - indexing_status["start_time"]
    
    return response

# コードベースに対して質問するエンドポイント
@app.post("/query", response_model=QueryResponse)
async def query_code(request: QueryRequest):
    try:
        # code_query.pyからquery_code関数をインポート
        from code_query import query_code
        
        # 質問を処理
        result = query_code(request.question)
        
        # レスポンスを整形
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "file": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return {"answer": result["result"], "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"クエリ処理中にエラーが発生しました: {str(e)}")

# 画像をアップロードして情報を抽出するエンドポイント
@app.post("/process_image", response_model=Dict[str, Any])
async def process_image(file: UploadFile = File(...), question: str = Form(None)):
    if not HAS_IMAGE_PROCESSING:
        return JSONResponse(
            status_code=501,
            content={"error": "画像処理ライブラリがインストールされていません。Dockerfileに必要なライブラリを追加してください。"}
        )
    
    try:
        # 画像を読み込む
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 画像からテキストを抽出
        extracted_text = pytesseract.image_to_string(image, lang='jpn+eng')
        
        # 画像を保存
        image_path = os.path.join("static", "images", file.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(contents)
        
        result = {
            "filename": file.filename,
            "extracted_text": extracted_text,
            "image_path": f"/static/images/{file.filename}"
        }
        
        # 質問がある場合、LLMを使用して回答
        if question:
            # code_query.pyからquery_code関数をインポート
            from code_query import llm
            
            prompt = f"""
以下は画像から抽出されたテキストです:

{extracted_text}

質問: {question}

上記の抽出されたテキストに基づいて、質問に対する回答を日本語で提供してください。
テキストに関連する情報がない場合は、「画像から抽出されたテキストには関連情報がありません」と回答してください。
"""
            response = llm.invoke(prompt)
            result["answer"] = response.content
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"画像処理中にエラーが発生しました: {str(e)}")

# Base64エンコードされた画像データを処理するエンドポイント
@app.post("/process_image_base64", response_model=Dict[str, Any])
async def process_image_base64(request: ImageQueryRequest):
    if not HAS_IMAGE_PROCESSING:
        return JSONResponse(
            status_code=501,
            content={"error": "画像処理ライブラリがインストールされていません。Dockerfileに必要なライブラリを追加してください。"}
        )
    
    try:
        # Base64データをデコード
        image_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_data))
        
        # 画像からテキストを抽出
        extracted_text = pytesseract.image_to_string(image, lang='jpn+eng')
        
        # 一意のファイル名を生成
        import uuid
        filename = f"{uuid.uuid4()}.png"
        
        # 画像を保存
        image_path = os.path.join("static", "images", filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)
        
        result = {
            "filename": filename,
            "extracted_text": extracted_text,
            "image_path": f"/static/images/{filename}"
        }
        
        # 質問がある場合、LLMを使用して回答
        if request.question:
            # code_query.pyからquery_code関数をインポート
            from code_query import llm
            
            prompt = f"""
以下は画像から抽出されたテキストです:

{extracted_text}

質問: {request.question}

上記の抽出されたテキストに基づいて、質問に対する回答を日本語で提供してください。
テキストに関連する情報がない場合は、「画像から抽出されたテキストには関連情報がありません」と回答してください。
"""
            response = llm.invoke(prompt)
            result["answer"] = response.content
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"画像処理中にエラーが発生しました: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # HTMLページを返す
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>コードベースRAGシステム</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .endpoint {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .endpoint h3 {
                margin-top: 0;
            }
            textarea {
                width: 100%;
                height: 100px;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                resize: vertical;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 4px;
                min-height: 100px;
                background-color: #f9f9f9;
            }
            .source {
                margin-top: 10px;
                padding: 10px;
                background-color: #eee;
                border-radius: 4px;
                font-size: 14px;
            }
            .source-file {
                font-weight: bold;
                color: #333;
            }
            #indexButton {
                background-color: #2196F3;
            }
            #indexButton:hover {
                background-color: #0b7dda;
            }
            .loading {
                display: none;
                margin-left: 10px;
            }
            .status-indicator {
                display: inline-block;
                margin-left: 10px;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 14px;
            }
            .status-idle {
                background-color: #e0e0e0;
                color: #333;
            }
            .status-running {
                background-color: #fff9c4;
                color: #ff6f00;
            }
            .status-completed {
                background-color: #c8e6c9;
                color: #2e7d32;
            }
            .status-error {
                background-color: #ffcdd2;
                color: #c62828;
            }
            .uml-section {
                margin-top: 30px;
                padding: 15px;
                background-color: #f0f8ff;
                border-radius: 5px;
            }
            .uml-image {
                max-width: 100%;
                margin-top: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            #indexDetails {
                margin-top: 10px;
                display: none;
                font-size: 14px;
                color: #666;
            }
            .info-box {
                background-color: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 10px 15px;
                margin: 15px 0;
                border-radius: 0 4px 4px 0;
            }
        </style>
    </head>
    <body>
        <h1>コードベースRAGシステム</h1>
        
        <div class="endpoint">
            <h3>インデックス作成</h3>
            <p>コードベースのインデックスを作成します。新しいコードファイル、画像、PDFを追加した場合は再実行してください。</p>
            <div class="info-box">
                <p><strong>ドキュメント処理について:</strong></p>
                <p>画像ファイル（PNG、JPEG、GIF、BMP）は <code>source_code/docs/images/</code> に配置してください。</p>
                <p>PDFファイルは <code>source_code/docs/pdf/</code> に配置してください。</p>
                <p>ファイルを追加した後は、インデックスを再作成する必要があります。</p>
            </div>
            <button id="indexButton" onclick="createIndex()">インデックス作成</button>
            <span id="indexStatus" class="status-indicator status-idle">未作成</span>
            <div id="indexDetails"></div>
        </div>
        
        <div class="endpoint">
            <h3>コードベースへの質問</h3>
            <p>コードベース、画像、PDFの内容に関する質問を入力してください：</p>
            <textarea id="questionInput" placeholder="例: このコードベースの構造について説明してください"></textarea>
            <button onclick="askQuestion()">送信</button>
            <span id="queryLoading" class="loading">処理中...</span>
        </div>
        
        <div id="result">
            <p>ここに回答が表示されます</p>
        </div>
        
        <script>
            // インデックス作成の状態を確認するための変数
            let indexingStatusChecker = null;
            
            // ページ読み込み時に初期状態を確認
            document.addEventListener('DOMContentLoaded', function() {
                checkIndexingStatus();
            });
            
            // インデックス作成の状態を確認する関数
            async function checkIndexingStatus() {
                try {
                    const response = await fetch('/index/status');
                    const data = await response.json();
                    
                    updateIndexingStatusUI(data);
                    
                    // 実行中の場合は定期的に状態を確認
                    if (data.is_running && !indexingStatusChecker) {
                        indexingStatusChecker = setInterval(checkIndexingStatus, 5000);
                    } else if (!data.is_running && indexingStatusChecker) {
                        clearInterval(indexingStatusChecker);
                        indexingStatusChecker = null;
                    }
                } catch (error) {
                    console.error('インデックス状態の確認中にエラーが発生しました:', error);
                }
            }
            
            // インデックス作成の状態表示を更新する関数
            function updateIndexingStatusUI(data) {
                const indexStatus = document.getElementById('indexStatus');
                const indexButton = document.getElementById('indexButton');
                const indexDetails = document.getElementById('indexDetails');
                
                // ステータスクラスをリセット
                indexStatus.className = 'status-indicator';
                
                // 状態に応じてUIを更新
                switch (data.status) {
                    case 'idle':
                        // start_timeとend_timeがnullの場合は「未作成」、そうでない場合は「作成完了」と表示
                        if (data.start_time === null && data.end_time === null) {
                            indexStatus.textContent = '未作成';
                            indexStatus.classList.add('status-idle');
                            indexButton.disabled = false;
                            indexDetails.style.display = 'none';
                        } else {
                            indexStatus.textContent = '作成完了';
                            indexStatus.classList.add('status-completed');
                            indexButton.disabled = false;
                            indexDetails.style.display = 'block';
                            
                            // 所要時間を表示（もしあれば）
                            if (data.duration) {
                                indexDetails.textContent = `所要時間: ${formatTime(data.duration)}`;
                            }
                        }
                        break;
                    case 'running':
                        indexStatus.textContent = '作成中...';
                        indexStatus.classList.add('status-running');
                        indexButton.disabled = true;
                        indexDetails.style.display = 'block';
                        
                        // 経過時間を表示
                        if (data.start_time) {
                            const elapsedTime = Math.floor(data.duration || 0);
                            indexDetails.textContent = `経過時間: ${formatTime(elapsedTime)}`;
                        }
                        break;
                    case 'completed':
                        indexStatus.textContent = '作成完了';
                        indexStatus.classList.add('status-completed');
                        indexButton.disabled = false;
                        indexDetails.style.display = 'block';
                        
                        // 所要時間を表示
                        if (data.duration) {
                            indexDetails.textContent = `所要時間: ${formatTime(data.duration)}`;
                        }
                        break;
                    case 'error':
                        indexStatus.textContent = 'エラー';
                        indexStatus.classList.add('status-error');
                        indexButton.disabled = false;
                        indexDetails.style.display = 'block';
                        indexDetails.textContent = data.message;
                        break;
                }
            }
            
            // 時間をフォーマットする関数（秒を分:秒形式に変換）
            function formatTime(seconds) {
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = Math.floor(seconds % 60);
                return `${minutes}分${remainingSeconds}秒`;
            }
            
            // インデックス作成を開始する関数
            async function createIndex() {
                const indexButton = document.getElementById('indexButton');
                const indexStatus = document.getElementById('indexStatus');
                const indexDetails = document.getElementById('indexDetails');
                
                indexButton.disabled = true;
                indexStatus.className = 'status-indicator status-running';
                indexStatus.textContent = "実行中...";
                
                try {
                    const response = await fetch('/index', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    const data = await response.json();
                    
                    // 状態確認を開始
                    if (!indexingStatusChecker) {
                        indexingStatusChecker = setInterval(checkIndexingStatus, 5000);
                    }
                } catch (error) {
                    indexStatus.className = 'status-indicator status-error';
                    indexStatus.textContent = "エラー";
                    indexDetails.style.display = 'block';
                    indexDetails.textContent = "エラーが発生しました: " + error.message;
                    indexButton.disabled = false;
                }
            }
            
            // 質問を送信する関数
            async function askQuestion() {
                const questionInput = document.getElementById('questionInput');
                const resultDiv = document.getElementById('result');
                const queryLoading = document.getElementById('queryLoading');
                
                if (!questionInput.value.trim()) {
                    resultDiv.innerHTML = "<p>質問を入力してください</p>";
                    return;
                }
                
                queryLoading.style.display = 'inline';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            question: questionInput.value
                        })
                    });
                    
                    const data = await response.json();
                    
                    let resultHTML = `<h3>回答:</h3><p>${data.answer.replace(/\\n/g, '<br>')}</p>`;
                    
                    if (data.sources && data.sources.length > 0) {
                        resultHTML += `<h3>参照ソース:</h3>`;
                        data.sources.forEach((source, index) => {
                            resultHTML += `
                                <div class="source">
                                    <div class="source-file">ファイル: ${source.file}</div>
                                    <pre>${source.content}</pre>
                                </div>
                            `;
                        });
                    }
                    
                    resultDiv.innerHTML = resultHTML;
                } catch (error) {
                    resultDiv.innerHTML = `<p>エラーが発生しました: ${error.message}</p>`;
                } finally {
                    queryLoading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # templatesディレクトリが存在しない場合は作成
    os.makedirs("templates", exist_ok=True)
    # 静的ファイルディレクトリが存在しない場合は作成
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 