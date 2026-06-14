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
├── .env.example          # API Key 配置模板
├── requirements.txt      # Python 依赖
├── RAG/                              # 方案一：RAG 检索增强路线
│   ├── rag_2.ipynb
│   └── rga.ipynb
├── Prompt-Engineering/               # 方案二：提示词工程路线
│   ├── one_shot_ner_re.ipynb
│   ├── one_shot_ner_re2v2pro.ipynb
│   └── few_shot_ner_re.ipynb
├── utils/                           # 工具脚本（清理后）
│   ├── split_data.py                # 数据切分（train/val）
│   ├── build_index.py               # FAISS 向量索引构建
│   ├── predict_rag.py              # RAG 预测器（检索+LLM）
│   ├── predict_prompt.py           # Prompt Engineering 预测器
│   └── convert_format.py          # 格式转换（模型输出→提交格式）
└── data/                           # 比赛数据（不提交）
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

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
# 然后编辑 .env 文件，填入你的 API Key
```

支持以下 API：
- DeepSeek（推荐）：`DEEPSEEK_API_KEY`
- 通义千问（阿里云百炼）：`DASHSCOPE_API_KEY`
- 智谱 AI（GLM）：`ZHIPU_API_KEY`

### 3. 数据切分

```bash
python utils/split_data.py --input data/train.json --output-dir data/
```

### 4. 构建 FAISS 索引（RAG 方案）

```bash
python utils/build_index.py --input data/train_data.json --output-dir data/
```

### 5. 批量预测

**RAG 方案**：
```bash
python utils/predict_rag.py \
  --test-file data/test_A.json \
  --output data/test_A_predictions.json \
  --api-key $DEEPSEEK_API_KEY \
  --base-url https://api.deepseek.com \
  --model deepseek-v4-flash
```

**Prompt Engineering 方案**：
```bash
python utils/predict_prompt.py \
  --test-file data/test_A.json \
  --output data/test_A_predictions.json \
  --api-key $DEEPSEEK_API_KEY
```

### 6. 格式转换（如需）

```bash
python utils/convert_format.py \
  --input data/test_A_predictions.json \
  --output data/submit.json
```

## 环境

- Python 3.10+
- 句向量模型：`all-MiniLM-L6-v2`（自动下载）
- LLM：DeepSeek V4 Flash / Qwen2.5-72B（通过 API 调用）

## License

MIT
