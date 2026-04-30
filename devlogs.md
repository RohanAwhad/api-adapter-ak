# Devlogs: IFBench qwen3-4b Thinking Mode Evaluation

## 2026-04-30: Notebook to script conversion

### What was done
- Converted `notebooks/ifbench/eda_train_data.ipynb` to `scripts/ifbench/eda_train_data.py`
- Kept only the qwen3-4b thinking mode path (removed filler/exploratory code)
- Script outputs:
  - JSONL data: `data/ifbench/subset_train_dataset_with_its_groups_gepa_prompt_think_qwen3-4b.jsonl`
  - Pass@N metrics: `results/ifbench/qwen3_4b_thinking/pass_rate.txt`
  - Advantages density plot: `results/ifbench/qwen3_4b_thinking/advantages_density.png`

### Key decisions
- **GROUP_SIZE=64** (notebook code had 32, but notebook's analysis data had 64 rollouts per group; 64 gives meaningful pass@64 metrics)
- **Kept pass@N logic as-is** from notebook (`if i <= n` — checks first n+1 rollouts, matching original behavior)
- **Kept bare excepts** in reward_fn and extract_adapter_response to preserve notebook behavior
- **Removed unused helpers** (`get_lgtm_count`, `get_fixme_count`) — never called in the pipeline
- **Import path**: `api_adapter.ifbench.eval_utils` (not `src.api_adapter...`) — matches other scripts in `scripts/ifbench/`

### 10-sample test results
- Pipeline runs end-to-end successfully (~6 min for 10 samples x 64 rollouts)
- Pass@N: 0.3 across all N (expected — small sample, consistent successes)
- Advantages density plot generated correctly with spread from -1.0 to +0.5
- JSONL data saved with all columns (groups, group_rewards, group_pass_at_n, advantages)

### 500-sample run
- Kicked off full run, estimated ~5-6 hours based on 10-sample timing
- Pending: collect results, commit, and raise PR
