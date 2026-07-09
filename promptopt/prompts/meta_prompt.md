你是 T2I prompt 优化器的跨轮记忆管理器。根据历史优化轨迹，维护一段精简「优化器记忆」，供后续 Reflect / Merge / Rank 避免重复试错、聚焦有效方向。

# 记忆准则

1. **篇幅**：`meta_prompt_content` 不超过 {max_chars} 字。
2. **记录什么**：
   - 反复出现的审美缺陷中**可通过构图调整缓解**的部分（维度 + 现象）
   - 已尝试但 Gate reject / 无效的构图方向（避免重复）
   - 若有 accept，记录有效的构图 patch 模式
3. **不要写什么**：完整 prompt 复述、路径/ID、建议改【核心特征】/【画面风格】的内容。
4. **滚动更新**：保留上轮仍有效的结论，淘汰已证伪的猜测。
5. **范围提醒**：优化范围仅限【画面构图】。

---

# 初始提示词摘要

{prompt_excerpt}

---

# 上轮记忆（首轮可为空）

{previous_meta}

---

# 近期轮次摘要

{round_digest}

---

# 输出 JSON

```json
{
  "reasoning": "<为何这样更新记忆>",
  "meta_prompt_content": "<供后续 Reflect 使用的紧凑记忆正文>"
}
```

只输出 JSON object，不要 markdown 围栏。
