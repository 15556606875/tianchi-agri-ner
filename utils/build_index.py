"""
FAISS 向量索引构建工具
从训练数据生成句子向量并构建 FAISS 索引，用于 RAG 检索
"""
import json
import os
import argparse
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


def build_index(json_path: str, output_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
    """
    构建 FAISS 索引
    
    Args:
        json_path: 训练数据 JSON 文件路径（如 train_data.json）
        output_dir: 输出目录（索引文件保存位置）
        model_name: 句向量模型名称
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("📂 加载数据...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   共 {len(data)} 条记录")

    # 2. 提取文本
    texts = [item['text'] for item in data if 'text' in item]
    print(f"   提取 {len(texts)} 条文本用于向量化")

    # 3. 加载模型
    print(f"🤖 加载模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 4. 生成向量
    print("🔢 生成向量...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embeddings = embeddings.astype('float32')
    print(f"   向量形状: {embeddings.shape}")

    # 5. 构建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"   FAISS 索引已构建，共 {index.ntotal} 个向量")

    # 6. 保存
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"💾 FAISS 索引: {index_path}")

    original_data_path = os.path.join(output_dir, "original_data.pkl")
    with open(original_data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 原始数据: {original_data_path}")

    text_list_path = os.path.join(output_dir, "text_list.pkl")
    with open(text_list_path, 'wb') as f:
        pickle.dump(texts, f)
    print(f"💾 文本列表: {text_list_path}")

    print("\n✅ 索引构建完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 FAISS 向量索引")
    parser.add_argument("--input", type=str, required=True, help="训练数据 JSON 路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="句向量模型")
    args = parser.parse_args()

    build_index(args.input, args.output_dir, args.model)
