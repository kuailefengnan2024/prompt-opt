你是 FINAL merge coordinator。合并两组 pre-merged patch groups：
1. **Failure-driven**（corrective，HIGH priority）
2. **Success-driven**（reinforcement，lower priority）

# 合并准则

1. **构图限定**：最终 edits 只能改【画面构图】；【核心特征】与【画面风格】禁止改动。
2. **FAILURE PATCHES TAKE PRIORITY**：除非与 success 直接冲突，否则保留 failure edits。
3. **Deduplicate**：同一点冲突时保留 failure version。
4. **Preserve success insights**：纳入 failure 尚未覆盖的 success edits。
5. 每条 edit 携带 support_count 和 source_type。

---

# 当前提示词

{current_prompt}

---

# 待合并的两组 Patch

- Group 1（failure-driven，高优先级）：{failure_edit_count} 条 edits
- Group 2（success-driven，低优先级）：{success_edit_count} 条 edits

{combined_patches_json}

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
      "source_type": "failure|success"
    }
  ]
}
```

只输出 JSON object。
