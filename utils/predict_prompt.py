"""
Prompt Engineering NER/RE 预测器
不使用 RAG 检索，直接使用 Few-Shot 示例 + LLM 进行预测
支持任意 OpenAI 兼容的 API
"""
import json
import os
import argparse
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI


# =============== 农业 NER 实体和关系定义 ===============
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
    "BIS": ("生物胁迫", "Biotic Stress")
}

RE_RELATIONS = {
    "CON": {"name": "包含", "desc": "品种属于某作物", "head": ["VAR"], "tail": ["CROP"]},
    "USE": {"name": "采用", "desc": "品种采用某种育种方法", "head": ["VAR"], "tail": ["BM"]},
    "HAS": {"name": "具有", "desc": "品种具备或关注某性状", "head": ["VAR"], "tail": ["TRT"]},
    "AFF": {"name": "影响", "desc": "非生物胁迫、基因、分子标记或 QTL 影响性状", "head": ["ABS", "GENE", "MRK", "QTL"], "tail": ["TRT"]},
    "OCI": {"name": "发生于", "desc": "性状或胁迫发生于某生育时期", "head": ["TRT", "ABS", "BIS"], "tail": ["GST"]},
    "LOI": {"name": "定位于", "desc": "分子标记、QTL 或基因定位于染色体或区间", "head": ["MRK", "QTL", "GENE"], "tail": ["CHR", "QTL"]}
}


def load_few_shot_examples(examples_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    加载 Few-Shot 示例
    
    Args:
        examples_path: 示例 JSON 文件路径（若为空则使用示例模板）
    """
    if examples_path and os.path.exists(examples_path):
        with open(examples_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 默认示例（可手动扩充）
    return [
        {
            "text": "ABA Insensitive 5 (ABI5) is a basic leucine zipper transcription factor.",
            "entities": [
                {"start": 0, "end": 17, "text": "ABA Insensitive 5", "label": "GENE"},
                {"start": 19, "end": 23, "text": "ABI5", "label": "GENE"}
            ],
            "relations": [
                {"head": "ABA Insensitive 5", "head_start": 0, "head_end": 17, "head_type": "GENE",
                 "tail": "ABI5", "tail_start": 19, "tail_end": 23, "tail_type": "GENE",
                 "label": "CON"}
            ]
        },
        {
            "text": "FtbHLH2 localizes in the nucleus. Its overexpression in Arabidopsis increases cold tolerance.",
            "entities": [
                {"start": 0, "end": 7, "text": "FtbHLH2", "label": "GENE"},
                {"start": 56, "end": 67, "text": "Arabidopsis", "label": "CROP"},
                {"start": 78, "end": 92, "text": "cold tolerance", "label": "TRT"}
            ],
            "relations": [
                {"head": "FtbHLH2", "head_start": 0, "head_end": 7, "head_type": "GENE",
                 "tail": "cold tolerance", "tail_start": 78, "tail_end": 92, "tail_type": "TRT",
                 "label": "AFF"}
            ]
        }
    ]


def build_prompt(input_text: str, examples: List[Dict[str, Any]]) -> str:
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
        '- relations: [{"head": str, "head_start": int, "head_end": int, "head_type": str, '
        '"tail": str, "tail_start": int, "tail_end": int, "tail_type": str, "label": str}]',
        '- 所有位置为字符级偏移（0-indexed），且必须在文本长度内。',
        '- 若无法识别，返回空列表 []。\n'
    ])
    
    for i, ex in enumerate(examples[:3], 1):
        lines.extend([
            f"\n示例 {i}：",
            f"文本：\"{ex['text']}\"",
            "输出：```json",
            json.dumps({"text": ex['text'], "entities": ex.get("entities", []), "relations": ex.get("relations", [])},
                       ensure_ascii=False, indent=2),
            "```"
        ])
    
    lines.extend([
        "\n=== 待处理文本 ===",
        f"文本：\"{input_text}\"\n",
        "请严格按上述格式输出 JSON："
    ])
    
    return "\n".join(lines)


def call_llm(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """调用 LLM（带重试）"""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
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
            
            result = json.loads(content)
            
            # 确保格式正确
            if "entities" not in result:
                result["entities"] = []
            if "relations" not in result:
                result["relations"] = []
                
            return result
            
        except Exception as e:
            print(f"[重试 {attempt}/{max_retries}] 异常: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue
    
    return {"entities": [], "relations": [], "error": "LLM call failed"}


def predict_single(client: OpenAI, model: str, input_text: str, 
                  examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """预测单条文本"""
    if not input_text or not input_text.strip():
        return {"text": input_text, "entities": [], "relations": []}
    
    prompt = build_prompt(input_text, examples)
    prediction = call_llm(client, model, prompt)
    
    return {
        "text": input_text,
        "entities": prediction.get("entities", []),
        "relations": prediction.get("relations", [])
    }


def batch_predict(test_file_path: str, api_key: str, base_url: str, model: str,
                 output_path: str = None, examples_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """批量预测"""
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"测试文件不存在: {test_file_path}")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    examples = load_few_shot_examples(examples_path)
    
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
                result = predict_single(client, model, text, examples)
            
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
    parser = argparse.ArgumentParser(description="Prompt Engineering NER/RE 批量预测")
    parser.add_argument("--test-file", type=str, required=True, help="测试文件 JSON 路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--examples", type=str, default=None, help="Few-Shot 示例 JSON 文件路径（可选）")
    parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY"), help="LLM API Key")
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
        output_path=args.output,
        examples_path=args.examples
    )
    
    print(f"\n📊 总计处理 {len(results)} 条，成功 {sum(1 for r in results if 'error' not in r)} 条")
