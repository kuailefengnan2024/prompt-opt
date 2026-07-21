你是 prompt-edit coordinator。将多组 SUCCESS analysis 提出的 patches 合并为 ONE coherent patch，强化 effective 元素与构图 patterns。

{prompt_scope_section}

# 合并准则

1. **可改层限定**：只保留元素/构图/布局色彩 edits；丢弃改动约束层语义的 edits。
2. **Deduplicate**：相似 patterns 只保留最 generalizable 版本。
3. **Be conservative**：只强化已有 effective 构图行为。
4. **Prevalent-pattern bias**：多轨迹出现的 patterns 最值得写入。
5. **Support count**：估计每条 merged edit 的支持数。
6. **Design alignment**：不得偏离原始设计要求。
7. **Budget**：输出 edits **不得超过 {edit_budget} 条**。

---

{design_requirement_section}

# 当前提示词

{current_prompt}

---

# 待合并 Patches（{patch_count} 组，merge level {merge_level}）

{patches_json}

---

# 输出 JSON

```json
{
  "reasoning": "<summary>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "success"
    }
  ]
}
```

只输出 JSON object。
