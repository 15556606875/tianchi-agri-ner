"""
格式转换工具
将模型预测输出转换为提交格式（图片格式）
"""
import json
import os
import argparse
from typing import List, Dict, Any, Optional


def convert_to_submission_format(model_results: List[Dict[str, Any]], 
                                 input_texts: List[str] = None) -> List[Dict[str, Any]]:
    """
    将模型预测的 {"entities": [...], "relations": [...]} 格式
    转换为提交格式 [{"text": "...", "entities": [...], "relations": [...]}]
    
    Args:
        model_results: 模型返回的结果列表
        input_texts: 原始输入文本列表（可选，用于填充 text 字段）
    
    Returns:
        转换后的列表格式数据
    """
    final_results = []
    
    for i, model_result in enumerate(model_results):
        # 获取原始文本
        text = ""
        if input_texts and i < len(input_texts):
            text = input_texts[i]
        elif "text" in model_result:
            text = model_result["text"]
        
        entities_out = []
        relations_out = []
        
        # 处理实体
        raw_entities = model_result.get("entities", [])
        entity_map = {}  # 用于关系验证
        
        for ent in raw_entities:
            try:
                start = int(ent.get("start", 0))
                end = int(ent.get("end", 0))
                ent_text = ent.get("text", "")
                label = ent.get("label", "")
                
                entity_obj = {
                    "start": start,
                    "end": end,
                    "text": ent_text,
                    "label": label
                }
                entities_out.append(entity_obj)
                
                # 存入映射表
                entity_map[ent_text] = {
                    "start": start,
                    "end": end,
                    "label": label
                }
            except Exception as e:
                print(f"警告：解析实体时出错 {ent}, 错误: {e}")
        
        # 处理关系
        raw_relations = model_result.get("relations", [])
        
        for rel in raw_relations:
            try:
                subject = rel.get("subject", rel.get("head", ""))
                relation_type = rel.get("relation_type", rel.get("label", ""))
                obj = rel.get("object", rel.get("tail", ""))
                
                # 查找头尾实体信息
                head_info = entity_map.get(subject)
                tail_info = entity_map.get(obj)
                
                if head_info and tail_info:
                    relation_obj = {
                        "head": subject,
                        "head_start": head_info["start"],
                        "head_end": head_info["end"],
                        "head_type": head_info["label"],
                        "tail": obj,
                        "tail_start": tail_info["start"],
                        "tail_end": tail_info["end"],
                        "tail_type": tail_info["label"],
                        "label": relation_type
                    }
                    relations_out.append(relation_obj)
                else:
                    missing = []
                    if not head_info:
                        missing.append(subject)
                    if not tail_info:
                        missing.append(obj)
                    print(f"警告：关系 {subject} -{relation_type}-> {obj} 中缺少实体: {missing}")
                    
            except Exception as e:
                print(f"警告：解析关系时出错 {rel}, 错误: {e}")
        
        # 组装最终结果
        final_result = {
            "text": text,
            "entities": entities_out,
            "relations": relations_out
        }
        
        final_results.append(final_result)
    
    return final_results


def batch_convert(input_file: str, output_file: str, text_file: Optional[str] = None):
    """
    批量转换文件
    
    Args:
        input_file: 模型输出 JSON 文件路径
        output_file: 转换后输出文件路径
        text_file: 原始文本文件路径（可选）
    """
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return
    
    # 加载模型输出
    with open(input_file, 'r', encoding='utf-8') as f:
        model_results = json.load(f)
    
    # 加载原始文本（可选）
    input_texts = None
    if text_file and os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            input_texts = json.load(f)
    
    # 转换
    print(f"正在转换 {len(model_results)} 条结果...")
    final_results = convert_to_submission_format(model_results, input_texts)
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成！结果已保存至: {output_file}")
    print(f"总共 {len(final_results)} 条数据")
    
    # 统计
    total_entities = sum(len(r["entities"]) for r in final_results)
    total_relations = sum(len(r["relations"]) for r in final_results)
    print(f"实体总数: {total_entities}")
    print(f"关系总数: {total_relations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="格式转换：模型输出 → 提交格式")
    parser.add_argument("--input", type=str, required=True, help="模型输出 JSON 文件")
    parser.add_argument("--output", type=str, required=True, help="转换后输出文件")
    parser.add_argument("--texts", type=str, default=None, help="原始文本文件（可选）")
    args = parser.parse_args()
    
    batch_convert(args.input, args.output, args.texts)
