# CS Demo 战术分析平台

> 面向 CS2 玩家与车队的 demo 战术分析系统：从 demo 解析到热力图、失守区域定位、个性化站位推荐，最终由 LLM 生成可读的战术报告。

**目标岗位方向**：大模型 / 数据科学 / 推荐系统
**当前状态**：阶段 0（启动中）

---

## 功能特性

- 🗺️ **走位热力图**：玩家/车队在特定地图上的站位分布（分 CT/T、分回合阶段）
- 💥 **交火 / 死亡热区**：击杀与死亡发生的空间分布
- 📊 **战术倾向分析**：开局路线、进攻/防守风格、回合节奏
- 🚨 **失守区域定位**：经常先丢人、导致回合失败的区域
- 🎯 **个性化站位推荐**：基于历史数据的混合推荐 + Learning to Rank
- 🤖 **LLM 战术报告**：大模型将结构化分析转化为自然语言解读

---

## 技术栈

| 类别 | 技术 |
|------|------|
| Demo 解析 | [awpy](https://github.com/pnxenopoulos/awpy) 3.x（Rust 后端 + Polars） |
| 数据处理 | polars、numpy、pandas |
| 可视化 | matplotlib、seaborn、plotly |
| 机器学习 | scikit-learn、LightGBM |
| 存储 | Parquet（polars 原生写出） |
| 服务 / 展示 | FastAPI、Streamlit |
| LLM | OpenAI / DeepSeek API |

> **awpy 3.x 亮点**：除了解析 demo，还提供 `NavMesh`（寻路）和 `VisibilityChecker`（视野判断），可直接用于"站位合理性"评估。

---

## 系统架构

```
数据源（个人 demo / HLTV 职业 demo / 地图数据）
        │
        ▼
解析层（awpy → Polars DataFrame）
        │
        ▼
存储层（Parquet：frames / rounds / events / zones）
        │
        ▼
特征层（zone 离散化 / 打法向量 / 上下文特征）
        │
   ┌────┴────┬────────────┬────────────┐
   ▼         ▼            ▼            ▼
分析可视化   推荐系统      LLM 报告     服务展示
（热力图/聚类/（统计基线/CF/（结构化→    （FastAPI/
 失守区域）  LTR）       自然语言）     Streamlit）
```

---

## 目录结构

```
cs-demo-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/               # 文档（项目计划书、架构设计）
├── data/
│   ├── demos/          # 原始 demo（不入库）
│   ├── processed/      # 解析后的 Parquet
│   └── maps/           # 地图底图与 zone 定义
├── src/
│   ├── parser/         # demo 解析
│   ├── storage/        # 落库与查询
│   ├── features/       # 特征工程、zone 离散化
│   ├── analysis/       # 热力图、聚类、失守分析
│   ├── recsys/         # 站位推荐系统
│   ├── llm/            # LLM 报告生成
│   └── api/            # FastAPI 服务
├── notebooks/          # 探索性分析
├── tests/              # 单元测试
└── docker/             # 容器化
```

---

## 快速开始

### 环境要求
- Python **3.11+**（本项目已用 conda 建好 `cs-analytics` 环境：Python 3.12，awpy wheels 支持 3.11–3.15）

### 激活环境（每次开发前）

```bash
conda activate cs-analytics
```

### 安装依赖（首次已装好，新机器需重跑）

```bash
# 注意：Windows 上请用 python -m pip（直接 pip 可能 Permission denied）
python -m pip install -r requirements.txt
```

> 可选：awpy 官方推荐使用 [uv](https://docs.astral.sh/uv/)：
> ```bash
> uv venv && uv pip install -r requirements.txt
> ```

### 解析第一个 demo

```bash
conda activate cs-analytics
python -m src.parser.parse "data/demos/my demo/xxx.dem" -o data/processed
```

解析后会在 `data/processed/<demo名>/` 下生成 12 张 Parquet 表（见 `docs/数据字典.md`）。

### 生成热力图（阶段 1）

```bash
python -m src.analysis.heatmap --data data/processed/<demo名>            # 全员走位
python -m src.analysis.heatmap --data data/processed/<demo名> --side CT  # 只看 CT
python -m src.analysis.heatmap --data data/processed/<demo名> --side T   # 只看 T
python -m src.analysis.heatmap --data data/processed/<demo名> --player 昵称  # 只看某玩家
python -m src.analysis.heatmap --data data/processed/<demo名> --table kills --role victim  # 死亡热区
```

---

## 路线图

| 阶段 | 内容 | 时间 |
|------|------|------|
| 0 | 跑通最小管道（解析 → 结构化数据） | 现在 ~ 12 月 |
| — | **备考暂停** | 12 月 ~ 1 月初 |
| 1 | 热力图与基础统计 | 1 月中 |
| 2 | 战术模式挖掘（聚类 / 失守区域） | 1 月底 ~ 2 月 |
| 3 | 站位推荐系统（核心） | 2 ~ 3 月 |
| 4 | LLM 战术报告 | 3 月 |
| 5 | 产品化与工程规范 | 持续 |

---

## 推荐系统设计（核心亮点）

将"站位建议"形式化为推荐问题：

| 术语 | 映射 |
|------|------|
| User | 队伍 / 玩家（角色 + 打法特征） |
| Item | 候选站位（zone / anchor 点） |
| Context | 地图、阵营、回合类型、回合阶段、人数差、经济 |
| Rating | 该站位在相似情境下的历史表现（胜率贡献、生存率、击杀交换比） |

三层方法：**统计基线 → 协同过滤 + 内容混合 → Learning to Rank**（离线评估用 NDCG / precision@k）。

---

## 数据来源

- 个人比赛 demo（Faceit / 官匹，需开启 demo 下载）
- [HLTV](https://www.hltv.org/) 职业比赛 demo（免费下载，作为"高水平基准"）

---

## 已知局限

- 推荐系统无法在线验证，离线评估只能证明"与历史高胜率站位的一致性"
- awpy 3.x 仅支持 CS2（Source 2），不支持 CS:GO（Source 1）

---

## License

MIT
