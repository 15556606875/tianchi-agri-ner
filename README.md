# 天池农业命名实体识别 (Tianchi Agri NER)

本仓库记录了我参加[天池大赛 - 农业命名实体识别](https://tianchi.aliyun.com/) 比赛的解题过程与代码。

## 项目背景

比赛任务：识别农业领域文本中的命名实体（如作物、品种、病虫害、农药、肥料等实体）。
评测指标：F1 分数（基于实体边界与类型匹配的严格匹配）。

## 目录结构

```
tianchi-agri-ner/
├── README.md
├── .gitignore
├── RAG/                              # 方案一：RAG 检索增强路线
│   ├── rag_2.ipynb
│   └── rga.ipynb
└── Prompt-Engineering/               # 方案二：提示词工程路线
    ├── one_shot_ner_re.ipynb
    ├── one_shot_ner_re2v2pro.ipynb
    └── few_shot_ner_re.ipynb
```

## 方案对比

| 方案 | 思路 | 最佳成绩 (F1) | 优点 | 缺点 |
|------|------|------|------|------|
| RAG 检索增强 | 检索相似样本 + LLM 生成 | 0.354 | 缓解 LLM 幻觉，结合外部知识 | 依赖检索质量，pipeline 复杂 |
| Prompt Engineering | One-shot / Few-shot 提示词 | 0.3500 | 实现简单，可解释性强 | 依赖模型能力，上限受限于大模型 |

## 关键发现

1. **大模型选择**：通义千问 Qwen 系列在中文农业 NER 任务上表现稳定
2. **Prompt 设计**：结构化输出（JSON Schema）+ 明确实体类型定义 + 边界示例，能显著提升 F1
3. **RAG 的作用**：当样本中包含稀有实体时，RAG 通过检索相似样例能有效提升召回率

## 复现方法

1. 准备天池比赛数据（报名比赛后从比赛页下载 `train.json`、`testA.json`）
2. 将数据放入 `data/` 目录（已被 `.gitignore` 过滤）
3. 按顺序运行各 notebook

## 环境

- Python 3.10+
- jupyter / notebook
- 阿里云 DashScope SDK（用于调用通义千问）

## License

MIT
