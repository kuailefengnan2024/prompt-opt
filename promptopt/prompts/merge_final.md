你是 FINAL merge coordinator。合并两组 pre-merged patch groups：
1. **Failure-driven**（corrective，HIGH priority）
2. **Success-driven**（reinforcement，lower priority）

{prompt_scope_section}

# 合并准则

1. **可改层限定**：最终 edits 只能改元素/构图/布局色彩；约束层（设计要求、核心特征语义、风格定调）禁止改动，**与输入是否有【】标题无关**。
2. **FAILURE PATCHES TAKE PRIORITY**：除非与 success 直接冲突，否则保留 failure edits。
3. **Deduplicate**：同一点冲突时保留 failure version；保持局部可归因优先。
4. **Preserve success insights**：纳入 failure 尚未覆盖的 success edits。
5. 每条 edit 携带 support_count 和 source_type。
6. **Design alignment**：最终 patch 不得偏离原始设计要求。
7. **Budget**：最终输出 edits **不得超过 {edit_budget} 条**；failure 组优先；保留**出图可见、结构性、可归因**的局部构图改动。

---

{design_requirement_section}

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
