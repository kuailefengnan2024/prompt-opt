你将收到来自一个 minibatch 的多条失败 agent trajectories，以及当前的 prompt document。

请将这些 trajectories 中的经验总结为一个完整的 replacement prompt document。

基于 minibatch rewrite 时，应以当前 trajectories 作为更新的主要 evidence。
保留必要的 task-format instructions，但避免机械保留 stale、redundant 或 conflicting rules。
相比带有弱证据 guidance 的冗长文档，优先给出精炼、连贯的 replacement prompt。

不要包含 task-specific answers、IDs、file paths、gold values 或 entity names。
如果 prompt 中包含位于 <!-- SLOW_UPDATE_START --> 和
<!-- SLOW_UPDATE_END --> 之间的 protected block，请保持该 block 不变。

只输出一个有效 JSON object：
{
  "batch_size": <number of trajectories analysed>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
  "patch": {
    "reasoning": "<brief summary of the rewrite>",
    "prompt_candidates": [
      {
        "title": "<short title>",
        "change_summary": ["<short change 1>", "<short change 2>"],
        "new_prompt": "<complete rewritten prompt document>"
      }
    ]
  }
}

在 "prompt_candidates" 中恰好返回一个 item。
