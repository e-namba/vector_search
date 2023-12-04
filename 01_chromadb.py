import chromadb  # chromadbライブラリをインポート
import openai  # OpenAIライブラリをインポート
from dotenv import load_dotenv  # 環境変数を扱うためのdotenvライブラリからload_dotenv関数をインポート
load_dotenv()  # .envファイルから環境変数を読み込む
import os  # OS関連の操作を行うためのライブラリをインポート
import pandas as pd  # データ分析ライブラリpandasをpdという名前でインポート
import datetime as dt  # 日付と時間を扱うためのdatetimeライブラリをdtという名前でインポート
from chromadb.utils import embedding_functions  # chromadbの埋め込み関数ユーティリティをインポート

API_KEY = os.getenv('OAI_API_KEY')  # 環境変数からOpenAIのAPIキーを取得
DATA_LENGTH = 1000  # 処理するデータの長さを指定

df = pd.read_csv('./data/saigai_data.csv')  # CSVファイルを読み込んでDataFrameを作成
# DataFrameの特定の列のデータ型をInt64に変換
df = df.astype({'年': 'Int64', '月': 'Int64', '業種（中分類）': 'Int64', '業種（小分類）': 'Int64', '事故の型': 'Int64', '起因物（大分類）.2': 'Int64', '起因物（中分類）.2': 'Int64', '起因物（小分類）.2': 'Int64'})
df_new = df[['年号','年','月','災害状況','業種（大分類）.1']]  # 必要な列だけを選択して新しいDataFrameを作成

client = chromadb.Client()  # chromadbのクライアントを作成
# OpenAIの埋め込み関数を設定
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=API_KEY,
                model_name="text-embedding-ada-002"
            )
# 新しいコレクションを作成し、埋め込み関数を指定
collection = client.create_collection(name="disaster_collection",embedding_function=openai_ef)

df_dic = df_new[:DATA_LENGTH].to_dict()  # DataFrameを辞書に変換し、指定されたデータ長さまでのデータを取得
docs = list(df_dic['災害状況'].values())  # '災害状況'列の値をリストとして取得
# '災害状況'キーを除いた辞書を作成
original_data = {key: value for key, value in df_dic.items() if key != '災害状況'}
# 各行のメタデータを辞書のリストとして作成
metadata = [
    {key: value[index] for key, value in original_data.items()}
    for index in range(len(next(iter(original_data.values()))))
]

ids = ["id{}".format(i) for i in df_dic['災害状況'].keys()]  # 各ドキュメントに一意のIDを割り当て

# ドキュメント、メタデータ、IDをコレクションに追加
collection.add(
    documents=docs,
    metadatas=metadata,
    ids=ids
)
# コレクションをクエリして結果を取得
results = collection.query(
    query_texts=["荷下ろし作業"],
    n_results=10
)
print(results)  # 結果を出力