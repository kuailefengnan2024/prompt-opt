你将收到基于 failed trajectories 写出的完整 prompt candidates，以及当前 prompt document。

请将它们合并为一个完整的 replacement prompt document。

合并 full-prompt candidates 时，保留必要的 task-format instructions，
但不要机械保留 stale、redundant 或 conflicting rules。
如果 candidates 之间不一致，优先选择更精炼、trajectory support 更清晰、
且与 replacement prompt 更一致的规则。

不要包含 task-specific answers、IDs、file paths、gold values 或 entity names。
如果当前 prompt 包含位于 <!-- SLOW_UPDATE_START --> 和
<!-- SLOW_UPDATE_END --> 之间的 protected block，请保持该 block 不变。

只输出一个有效 JSON object：
{
  "reasoning": "<brief summary of how the candidates were combined>",
  "prompt_candidates": [
    {
      "title": "<short title>",
      "change_summary": ["<short change 1>", "<short change 2>"],
      "new_prompt": "<complete merged prompt document>",
      "support_count": <integer>,
      "source_type": "failure"
    }
  ]
}

在 "prompt_candidates" 中恰好返回一个 item。
