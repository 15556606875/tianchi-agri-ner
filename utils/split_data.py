"""
数据切分工具：将训练集按 8:2 切分为训练集和验证集
"""
import json
import os
import argparse
from sklearn.model_selection import train_test_split


def split_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    切分 JSON 数据文件
    
    Args:
        input_path: 输入 JSON 文件路径（如 train.json）
        output_dir: 输出目录
        test_size: 验证集比例（默认 0.2）
        random_state: 随机种子（保证可复现）
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到文件: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    print(f"✅ 数据加载成功！共 {len(all_data)} 条")

    train_data, val_data = train_test_split(
        all_data, test_size=test_size, random_state=random_state
    )
    print(f"📋 训练集: {len(train_data)} 条")
    print(f"🧪 验证集: {len(val_data)} 条")

    train_output = os.path.join(output_dir, "train_data.json")
    val_output = os.path.join(output_dir, "val_data.json")

    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存: {train_output}")

    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存: {val_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="切分训练数据")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集比例")
    args = parser.parse_args()

    split_data(args.input, args.output_dir, args.test_size)
