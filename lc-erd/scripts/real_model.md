# Real-model LC-ERD checklist

This machine currently has PyTorch but not `transformers`. After installing the
optional dependencies, use this checklist to scale from the toy reproduction to
a real reasoning model.

## Data

- GSM8K: use final numeric-answer verifier.
- MATH-500: use final answer parser plus exact/sympy verifier where possible.
- Keep train/eval splits fixed and log random seeds.

## Exploration

- For every prompt, sample `K=64` trajectories from the frozen base model.
- Store:
  - prompt id;
  - generated text;
  - parsed step boundaries;
  - per-step action token ids;
  - per-step base logprob;
  - per-step logits, preferably top-k compressed.
- Retain only trajectories accepted by `Ver(tau)=1`.

## LLE cache

- Use `confidence_weights` with average log probability and temperature.
- Use `weighted_consensus_logits` to produce `pi_LLE`.
- For memory, pack logits with `TopKLogitCache(k=64)`.

## Optimization

- Initialize policy from the base model.
- For each verified trajectory step:
  - compute VLP with `beta=0.1`;
  - reconstruct dense LC-ERD rewards with `gamma=0.95`;
  - optimize TD loss plus LLE logit-consistency loss.
- Use target-network EMA (`xi`/`target_tau`) and gradient clipping.

## Evaluation

- Report success rate and a logic consistency score:

  ```text
  LCS = mean_h [1 - normalized_KL(pi_theta(.|s_h), pi_LLE(.|s_h))]
  ```

- Compare against base, SFT, GRPO-style uniform terminal reward, and
  LC-ERD without VLP.
