你是 prompt-edit coordinator。将多组 FAILURE analysis 提出的 patches 合并为 ONE coherent、non-redundant patch。

# 合并准则

1. **构图限定**：只保留作用于【画面构图】的 edits；丢弃任何会改动【核心特征】或【画面风格】的 edits。
2. **Deduplicate**：相似 edits 只保留措辞最佳版本。
3. **Resolve conflicts**：矛盾时选 justification 更强的一方。
4. **Preserve unique insights**：纳入所有 non-redundant corrective edits。
5. **Prevalent-pattern bias**：多 patch 重复出现的 edits 高优先级保留。
6. **Independence**：任意两条 edits 不得 target 同一 text region。
7. **Support count**：估计每条 merged edit 有多少 source patches 支持。

---

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
      "source_type": "failure"
    }
  ]
}
```

只输出 JSON object。
