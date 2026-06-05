# prompt-opt：T2I Prompt 优化（Direct）

Fork [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)，**Reflect + Gate**，**固定 brief → 最优 T2I prompt**。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**快照分支：** `backup/pre-t2i-direct`

---

## 术语

| 概念 | 键 / 产物 | 说明 |
|------|-----------|------|
| **prompt** | `prompt_init`、`prompts/prompt_vNNNN.md` | 优化对象，T2I 正文 |
| **best prompt** | `best_prompt.md` | 验收通过的历史最优 |
| **brief** | `data/.../*.json` | 固定业务约束，不改 |
| **rubric** | 适配器打分 | 审美维度 → `hard`/`soft` |
| **API 配置** | `.env`、`configs/` | 模型 endpoint，非 prompt 正文 |
| **train_runs** | `env.train_runs` | Rollout 重复出图（无 seed） |
| **gate_runs** | `env.gate_runs` | Gate 重复出图 |
| **optimizer 模板** | `promptopt/llm_templates/` | Reflect 用 LLM 系统提示，`load_template()` |

| YAML | 扁平 |
|------|------|
| `env.prompt_init` | `prompt_init` |
| `optimizer.prompt_update_mode` | `prompt_update_mode` |
| `optimizer.use_meta_prompt` | `use_meta_prompt` |
| `env.best_prompt_file` | `best_prompt_file` |

---

## 设计结论

| 议题 | 结论 |
|------|------|
| 模式 | **direct**：优化 prompt 正文 |
| meta | TODO |
| 随机性 | 无 seed；`train_runs` / `gate_runs` |
| 初始 prompt | 必填 `prompt_init` |
| 收敛 | 无 loss；跑满 epoch；看 `best_score` |

---

## Direct 流程

```mermaid
flowchart TB
    subgraph fixed ["固定"]
        BRIEF["brief"]
        API["API"]
        RUBRIC["rubric"]
    end
    P0["initial.md"] --> R1["① Rollout"]
    R1 --> R2["② Reflect"] --> R3["③④"] --> R4["⑤"]
    R4 --> G1["⑥ Gate"] --> G2{"更高?"}
    G2 -->|是| OK["更新 best"]
    G2 -->|否| NO["保留"]
    fixed -.-> R1
    fixed -.-> G1
    OK --> R1
    NO --> R1
    OK --> OUT["best_prompt.md"]
```

| 阶段 | 含义 |
|------|------|
| ① | prompt × `train_runs` → 均分 |
| ② | 低分 → patches |
| ③④ | 合并；`edit_budget` |
| ⑤ | 候选 prompt |
| ⑥ | × `gate_runs`；高于 current 才 accept |

---

## 目录

| 路径 | 职责 |
|------|------|
| `promptopt/` | 核心包 |
| `promptopt/envs/t2i/prompts/initial.md` | 初始 prompt |
| `promptopt/llm_templates/` | Optimizer LLM 模板 |
| `promptopt/engine/trainer.py` | 主循环 |
| `configs/t2i/default.yaml` | 配方 |

```
outputs/<run>/
├── best_prompt.md
├── prompts/prompt_v0001.md
└── steps/step_XXXX/
```

---

## 运行

```bash
pip install -e . && cp .env.example .env
python scripts/train.py --config configs/t2i/default.yaml \
  --split_dir data/t2i_split --out_root outputs/t2i_run
```

| CLI | 说明 |
|-----|------|
| `--prompt_init` | 初始 prompt |
| `--prompt_update_mode` | `patch` / `rewrite_from_suggestions` / … |
| `--use_meta_prompt` | 跨 epoch 优化器记忆 |

---

## 上游

[Microsoft SkillOpt](https://github.com/microsoft/SkillOpt) · `backup/archive/`
