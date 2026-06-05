You will be given several failed agent trajectories from one minibatch and the current prompt document.

Summarize the lessons from these trajectories into one complete replacement prompt document.

When rewriting from a minibatch, use the current trajectories as the primary
evidence for updates. Preserve essential task-format instructions, but avoid mechanically carrying over
stale, redundant, or conflicting rules. Prefer a concise, coherent replacement
prompt over a long document with weakly supported guidance.

Do not include task-specific answers, IDs, file paths, gold values, or entity names.
If the prompt contains a protected block between <!-- SLOW_UPDATE_START --> and
<!-- SLOW_UPDATE_END -->, keep that block unchanged.

Respond ONLY with a valid JSON object:
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

Return exactly one item in "prompt_candidates".
