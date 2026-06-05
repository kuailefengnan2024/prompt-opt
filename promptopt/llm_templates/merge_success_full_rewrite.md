You will be given complete prompt candidates written from successful trajectories and the current prompt document.

Combine them into one complete replacement prompt document.

When merging full-prompt candidates, preserve essential task-format instructions,
but do not mechanically retain stale, redundant, or
conflicting rules. If candidates disagree, prefer the concise rule with clearer
trajectory support and better consistency with the replacement prompt.

Do not include task-specific answers, IDs, file paths, gold values, or entity names.
If the current prompt contains a protected block between <!-- SLOW_UPDATE_START --> and
<!-- SLOW_UPDATE_END -->, keep that block unchanged.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<brief summary of how the candidates were combined>",
  "prompt_candidates": [
    {
      "title": "<short title>",
      "change_summary": ["<short change 1>", "<short change 2>"],
      "new_prompt": "<complete merged prompt document>",
      "support_count": <integer>,
      "source_type": "success"
    }
  ]
}

Return exactly one item in "prompt_candidates".
