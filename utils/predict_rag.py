"""
RAG NER/RE 预测器
使用 FAISS 检索相似样本 + LLM 进行命名实体识别和关系抽取
支持任意 OpenAI 兼容的 API（DeepSeek、Qwen 等）
"""
import json
import os
import time
import argparse
import pickle
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss


# =============== 农业 NER 实体类型（12类）===============
NER_TYPES = {
    "CROP": ("作物", "Crop"),
    "VAR": ("品种", "Variety/Cultivar"),
    "TRT": ("性状", "Trait"),
    "GST": ("生育时期", "Growth Stage"),
    "GENE": ("基因", "Gene"),
    "QTL": ("数量性状位点", "QTL"),
    "MRK": ("分子标记", "Molecular Marker"),
    "CHR": ("染色体", "Chromosome"),
    "BM": ("育种方法", "Breeding Method"),
    "CROSS": ("亲本/杂交组合", "Parent/Cross"),
    "ABS": ("非生物胁迫", "Abiotic Stress"),
    "BIS": ("生物胁迫", "Biotic Stress"),
}

# =============== 关系类型（6类）===============
RE_RELATIONS = {
    "CON": {"name": "包含", "desc": "品种属于某作物", "head": ["VAR"], "tail": ["CROP"]},
    "USE": {"name": "采用", "desc": "品种采用某种育种方法", "head": ["VAR"], "tail": ["BM"]},
    "HAS": {"name": "具有", "desc": "品种具备或关注某性状", "head": ["VAR"], "tail": ["TRT"]},
    "AFF": {"name": "影响", "desc": "非生物胁迫、基因、分子标记或 QTL 影响性状", "head": ["ABS", "GENE", "MRK", "QTL"], "tail": ["TRT"]},
    "OCI": {"name": "发生于", "desc": "性状或胁迫发生于某生育时期", "head": ["TRT", "ABS", "BIS"], "tail": ["GST"]},
    "LOI": {"name": "定位于", "desc": "分子标记、QTL 或基因定位于染色体或区间", "head": ["MRK", "QTL", "GENE"], "tail": ["CHR", "QTL"]},
}

VALID_ENTITY_LABELS = set(NER_TYPES.keys())
VALID_RELATION_LABELS = set(RE_RELATIONS.keys())


class RAGNERREPredictor:
    """RAG NER/RE 预测器"""

    def __init__(self, api_key: str, base_url: str, faiss_index_path: str, original_data_path: str, model: str = "deepseek-v4-flash"):
        """
        初始化预测器

        Args:
            api_key: LLM API Key（从环境变量读取）
            base_url: API 基础 URL（OpenAI 兼容）
            faiss_index_path: FAISS 索引文件路径
            original_data_path: 原始数据 pickle 文件路径
            model: 模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 加载向量模型与索引
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(faiss_index_path)

        # 加载原始示例数据
        with open(original_data_path, 'rb') as f:
            self.original_data = pickle.load(f)

    def _normalize_label(self, label: str, label_type: str) -> Optional[str]:
        """归一化标签"""
        if not label:
            return None
        label = label.strip().upper()

        if label_type == "entity":
            if label in VALID_ENTITY_LABELS:
                return label
            for k, (cn, en) in NER_TYPES.items():
                if label in [cn, en, cn.upper(), en.upper(), k]:
                    return k
        elif label_type == "relation":
            if label in VALID_RELATION_LABELS:
                return label
            for k, info in RE_RELATIONS.items():
                if label in [info["name"], info["name"].upper(), k]:
                    return k
        return None

    def _validate_entity(self, ent: Dict[str, Any], input_text: str) -> Optional[Dict[str, Any]]:
        """验证单个实体"""
        if not isinstance(ent, dict):
            return None

        start = ent.get("start")
        end = ent.get("end")
        text = ent.get("text")
        label = ent.get("label")

        if start is None or end is None or text is None or label is None:
            return None

        try:
            start, end = int(start), int(end)
        except (ValueError, TypeError):
            return None

        if not (0 <= start <= end <= len(input_text)):
            return None

        # 修正文本（以防模型输出的位置不准）
        if input_text[start:end] != text:
            text = input_text[start:end]

        norm_label = self._normalize_label(label, "entity")
        if not norm_label:
            return None

        return {"start": start, "end": end, "text": text, "label": norm_label}

    def _validate_relation(self, rel: Dict[str, Any], input_text: str) -> Optional[Dict[str, Any]]:
        """验证单个关系"""
        if not isinstance(rel, dict):
            return None

        required = ["head", "tail", "head_start", "head_end", "tail_start", "tail_end",
                     "head_type", "tail_type", "label"]
        if not all(k in rel for k in required):
            return None

        try:
            h_s, h_e = int(rel["head_start"]), int(rel["head_end"])
            t_s, t_e = int(rel["tail_start"]), int(rel["tail_end"])
        except (ValueError, TypeError):
            return None

        if not (0 <= h_s <= h_e <= len(input_text) and 0 <= t_s <= t_e <= len(input_text)):
            return None

        head_type = self._normalize_label(rel["head_type"], "entity")
        tail_type = self._normalize_label(rel["tail_type"], "entity")
        rel_label = self._normalize_label(rel["label"], "relation")

        if not head_type or not tail_type or not rel_label:
            return None

        # 检查关系类型约束
        rel_def = RE_RELATIONS[rel_label]
        if isinstance(rel_def["head"], list) and head_type not in rel_def["head"]:
            return None
        if isinstance(rel_def["tail"], list) and tail_type not in rel_def["tail"]:
            return None

        head_text = input_text[h_s:h_e]
        tail_text = input_text[t_s:t_e]

        return {
            "head": head_text, "head_start": h_s, "head_end": h_e, "head_type": head_type,
            "tail": tail_text, "tail_start": t_s, "tail_end": t_e, "tail_type": tail_type,
            "label": rel_label
        }

    def retrieve_similar_examples(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """检索相似示例"""
        if not query_text.strip():
            return []
        emb = self.embedding_model.encode([query_text], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(emb, k=top_k)
        retrieved = []
        for idx in indices[0]:
            if 0 <= idx < len(self.original_data):
                retrieved.append(self.original_data[idx])
        return retrieved

    def build_prompt(self, input_text: str, examples: List[Dict[str, Any]]) -> str:
        """构建提示词"""
        lines = [
            "你是一名农业遗传育种领域的专业信息抽取专家。",
            "请严格根据以下规范，从输入文本中识别命名实体（NER）和实体间关系（RE）：\n",
            "=== 实体类型（12类） ==="
        ]

        for lbl, (cn, en) in NER_TYPES.items():
            lines.append(f"- {lbl}: {cn} ({en})")

        lines.append("\n=== 关系类型（6类） ===")
        for rid, info in RE_RELATIONS.items():
            lines.append(f"- {rid} ({info['name']}): {info['desc']}")

        lines.extend([
            "\n=== 输出格式要求 ===",
            '- 仅输出一个 JSON 对象，包含 "text"、"entities" 和 "relations" 三个键，无任何额外说明。',
            '- entities: [{"start": int, "end": int, "text": str, "label": str}]',
            '- relations: [{"head": str, "head_start": int, "head_end": int, "head_type": str, "tail": str, "tail_start": int, "tail_end": int, "tail_type": str, "label": str}]',
            '- 所有位置为字符级偏移（0-indexed），且必须在文本长度内。',
            '- 若无法识别，返回空列表 []。\n'
        ])

        for i, ex in enumerate(examples[:3], 1):
            lines.extend([
                f"\n示例 {i}：",
                f"文本：\"{ex['text']}\"",
                "输出：```json",
                json.dumps({"text": ex['text'], "entities": ex.get("entities", []), "relations": ex.get("relations", [])}, ensure_ascii=False, indent=2),
                "```"
            ])

        lines.extend([
            "\n=== 待处理文本 ===",
            f"文本：\"{input_text}\"\n",
            "请严格按上述格式输出 JSON："
        ])

        return "\n".join(lines)

    def call_llm(self, prompt: str, input_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """调用 LLM（带重试）"""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                    temperature=0.05,
                    top_p=0.85,
                    seed=42
                )

                content = response.choices[0].message.content

                # 提取 JSON
                for delim in ["```json", "```"]:
                    if delim in content:
                        start = content.find(delim) + len(delim)
                        end = content.find("```", start)
                        if end == -1:
                            end = len(content)
                        content = content[start:end].strip()
                        break

                try:
                    raw_result = json.loads(content)
                except json.JSONDecodeError:
                    content = content.strip().rstrip(',').rstrip('}')
                    content = content.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
                    raw_result = json.loads(content)

                result = {"text": input_text, "entities": [], "relations": []}

                for ent in raw_result.get("entities", []):
                    validated = self._validate_entity(ent, input_text)
                    if validated:
                        result["entities"].append(validated)

                for rel in raw_result.get("relations", []):
                    validated = self._validate_relation(rel, result["entities"], input_text)
                    if validated:
                        result["relations"].append(validated)

                return result

            except Exception as e:
                print(f"[重试 {attempt}/{max_retries}] 异常: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

        print("[警告] LLM 调用全部失败，返回空结果")
        return {"text": input_text, "entities": [], "relations": []}

    def predict_single(self, input_text: str, top_k: int = 3) -> Dict[str, Any]:
        """预测单条文本"""
        if not input_text or not input_text.strip():
            return {"text": input_text, "entities": [], "relations": []}

        examples = self.retrieve_similar_examples(input_text, top_k=top_k)
        prompt = self.build_prompt(input_text, examples)
        prediction = self.call_llm(prompt, input_text=input_text)

        return prediction


def batch_predict(test_file_path: str, api_key: str, base_url: str, model: str,
                  output_path: str = None, batch_size: int = 1) -> List[Dict[str, Any]]:
    """
    批量预测

    Args:
        test_file_path: 测试文件 JSON 路径
        api_key: LLM API Key
        base_url: API 基础 URL
        model: 模型名称
        output_path: 输出文件路径
        batch_size: 批大小（当前串行，保留参数）
    """
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"测试文件不存在: {test_file_path}")

    base_dir = os.path.dirname(test_file_path)
    faiss_index_path = os.path.join(base_dir, "faiss_index.bin")
    original_data_path = os.path.join(base_dir, "original_data.pkl")

    for path in [faiss_index_path, original_data_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"依赖文件缺失: {path}")

    predictor = RAGNERREPredictor(api_key, base_url, faiss_index_path, original_data_path, model)

    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []
    print(f"开始预测 {len(test_data)} 条记录...")

    for i, item in enumerate(tqdm(test_data, desc="进度")):
        try:
            text = None
            if isinstance(item, dict):
                text = item.get("text") or item.get("Text") or item.get("sentence") or item.get("content")
                if not text:
                    keys = [k for k in item.keys() if k.lower() in ['text', 'content', 'abstract']]
                    if keys:
                        text = item[keys[0]]
            if not text:
                text = str(item)

            if not text.strip():
                result = {"text": "", "entities": [], "relations": [], "error": "空输入"}
            else:
                result = predictor.predict_single(text, top_k=3)

            results.append(result)

        except Exception as e:
            print(f"[错误] 第 {i+1} 条: {e}")
            results.append({"text": str(item)[:100] if item else "", "entities": [], "relations": [], "error": str(e)})

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 预测完成！结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG NER/RE 批量预测")
    parser.add_argument("--test-file", type=str, required=True, help="测试文件 JSON 路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY"), help="LLM API Key（或设环境变量 LLM_API_KEY）")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="API 基础 URL")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="模型名称")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("请设置 LLM_API_KEY 环境变量或 --api-key 参数")

    results = batch_predict(
        test_file_path=args.test_file,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        output_path=args.output
    )

    print(f"\n📊 总计处理 {len(results)} 条，成功 {sum(1 for r in results if 'error' not in r)} 条")
