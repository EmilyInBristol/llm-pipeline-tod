import argparse
import pickle
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from multiwoz_utils.data_loader import load_multiwoz
from multiwoz_utils.database import MultiWOZDatabase
from multiwoz_utils.dialog_iterator import iterate_dialogues


def main():
    parser = argparse.ArgumentParser(description="构建 MultiWOZ 向量数据库（vec 文件）")
    parser.add_argument('--database_path', type=str, default='./multiwoz_database', help='数据库json文件夹路径')
    parser.add_argument('--split', type=str, default='train', help='数据集 split，train/validation/test')
    parser.add_argument('--context_size', type=int, default=3, help='对话上下文轮数')
    parser.add_argument('--output', type=str, default='multiwoz-context-db.vec', help='输出 vec 文件名')
    parser.add_argument('--max_dialogs', type=int, default=50, help='每个领域最大对话数')
    args = parser.parse_args()

    print(f"加载 MultiWOZ 数据集 split={args.split} ...")
    data = load_multiwoz(args.split)
    print(f"共 {len(data)} 条对话")

    print(f"加载数据库: {args.database_path}")
    database = MultiWOZDatabase(args.database_path)

    print("初始化 embedding 模型: sentence-transformers/all-mpnet-base-v2 ...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print("生成结构化对话 turn ...")
    docs = []
    for turn in tqdm(iterate_dialogues(data, database, context_size=args.context_size, max_dialogs=args.max_dialogs), desc="Processing", unit="turn"):
        doc = Document(page_content=turn['page_content'], metadata=turn['metadata'])
        docs.append(doc)

    print(f"共生成 {len(docs)} 个对话 turn，开始构建 FAISS 向量库 ...")
    faiss_vs = FAISS.from_documents(documents=docs, embedding=embeddings)

    print(f"保存到 {args.output} ...")
    with open(args.output, 'wb') as f:
        pickle.dump(faiss_vs, f)
    print(f"已保存 FAISS 向量库到 {args.output}")

if __name__ == '__main__':
    main() 