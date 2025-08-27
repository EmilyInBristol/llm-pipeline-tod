import argparse
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from multiwoz_utils.data_loader import load_multiwoz
from multiwoz_utils.database import MultiWOZDatabase
from multiwoz_utils.dialog_iterator import iterate_dialogues


def main():
    parser = argparse.ArgumentParser(description="Build MultiWOZ vector database (vec file)")
    parser.add_argument('--database_path', type=str, default='./multiwoz_database', help='Database json folder path')
    parser.add_argument('--split', type=str, default='train', help='Dataset split, train/validation/test')
    parser.add_argument('--context_size', type=int, default=3, help='Dialogue context turns')
    parser.add_argument('--output', type=str, default='multiwoz-context-db.vec', help='Output vec filename')
    parser.add_argument('--max_dialogs', type=int, default=50, help='Maximum dialogues per domain')
    args = parser.parse_args()

    print(f"Loading MultiWOZ dataset split={args.split} ...")
    data = load_multiwoz(args.split)
    print(f"Total {len(data)} dialogues")

    print(f"Loading database: {args.database_path}")
    database = MultiWOZDatabase(args.database_path)

    print("Initializing embedding model: Qwen/Qwen3-Embedding-0.6B ...")
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

    print("Generating structured dialogue turns ...")
    docs = []
    for turn in tqdm(iterate_dialogues(data, database, context_size=args.context_size, max_dialogs=args.max_dialogs), desc="Processing", unit="turn"):
        if not turn['gt_state']:
            continue
        doc = Document(page_content=turn['page_content'], metadata=turn['metadata'])
        docs.append(doc)

    print(f"Generated {len(docs)} dialogue turns, starting to build FAISS vector store ...")
    faiss_vs = FAISS.from_documents(documents=docs, embedding=embeddings)

    print(f"Saving to {args.output} ...")
    with open(args.output, 'wb') as f:
        pickle.dump(faiss_vs, f)
    print(f"FAISS vector store saved to {args.output}")

if __name__ == '__main__':
    main() 