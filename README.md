# EVAPO

**Evidence-Validated Adaptive Policy Optimization with On-Policy Self-Distillation for Audio-Language Models**

[English](README.md) | [中文](README_zh.md)

EVAPO is an audio-grounded post-training framework for audio-language models. It combines answer-verified reinforcement learning with selective on-policy self-distillation: every valid example uses a verified answer as its optimization target, while the same model's privileged-context distribution is used only when independently verified evidence can correct the current mistake.

## Results

### Main results

| Model | MMAU test-mini | MMSU |
|---|---:|---:|
| Step-Audio-R1.1 | 77.7 | – |
| Step-Audio 2 | 78.0 | – |
| MiMo-Audio-7B-Instruct | 74.9 | – |
| Kimi-Audio | 65.2 | – |
| LongCat-Next | 76.4 | – |
| Qwen3-Omni-30B-A3B-Instruct | 77.5 | – |
| Gemini 3.1 Pro | 80.7* | 82.7* |
| Qwen3.5-Omni-Plus | 81.4* | 80.7* |
| FireRedAudio | 82.0 | 83.3 |
| EVAPO | **83.40%** | **83.88%** |

Note: Results marked with * are reproduced test results.

## Motivation

Audio-language models can often recognize acoustic content, but they still struggle to produce reliable answers grounded in that content. Post-training therefore faces three related problems:

- **Sparse RL supervision:** pure RL provides a trustworthy answer objective, but sparse rewards offer little direction for difficult audio errors.
- **Unreliable self-distillation signals:** standard OPSD provides dense guidance, but the privileged-context distribution may be unreliable on student-generated states and can reinforce incorrect or irrelevant behavior.
- **Privileged-information leakage:** privileged text or reasoning can improve the model's judgment under the privileged context, but it can also reveal answers or teach shortcuts unavailable at deployment.

### Design principle

An effective post-training recipe should keep the verified answer as the final training objective while using self-distillation only when it has demonstrated real corrective value. EVAPO therefore treats privileged-context supervision as a targeted correction rather than a universal imitation target, preserving the model's own audio-grounded decision making while providing direction where it is needed.

## Method

EVAPO connects answer verification, evidence gating, selective OPSD, and SAPO-based policy optimization in one training flow: it first checks whether candidate evidence can correct the model's current error, then applies privileged-context guidance only to the rollouts that need correction.

No separate teacher model is introduced. The student and privileged-context distributions are produced by the same model under different contexts.

### 1. Evidence verification and gating

Candidate evidence is evaluated through four complementary views: the original evidence, an answer-redacted evidence view, a shuffled-evidence control, and the ordinary student view. Evidence is admitted to the privileged self-distillation branch only when all of the following conditions hold:

1. the ordinary student view gives the wrong answer;
2. the redacted evidence view reaches the correct answer;
3. the shuffled-evidence control does not reach the same answer;
4. the evidence contains no gold answer, complete option, or other form of label leakage.

This gate tests whether the evidence has demonstrated corrective value instead of assuming that a longer rationale is useful.

### 2. Answer-verified SAPO objective

Every valid training example participates in SAPO-based policy optimization with the verified answer as its training objective. Only examples that pass the evidence gate expose a privileged `teacher_prompt` context to the self-distillation branch.

### 3. Selective directional OPSD

For gated examples, the log-probability gap between the privileged-context and ordinary student distributions provides a directional self-distillation signal. The intervention is restricted to negative or incorrect student rollouts and retains only gaps that point toward correction. This prevents the privileged context from overwriting behavior that is already correct and reduces the impact of noisy self-distillation signals.

## What the method addresses

EVAPO is designed to address four practical failure modes in audio post-training:

- **Noisy self-distillation feedback:** evidence must demonstrate that the privileged context corrects a real student failure.
- **Sparse RL supervision:** dense self-distillation direction is added only on verified rescue cases, while the answer reward remains the final objective.
- **Shortcut and answer leakage:** redaction and shuffled-evidence controls reject evidence that works only because it reveals the label.
- **Over-intervention:** negative-only directional updates leave correct student behavior under the answer objective.

The method is a training recipe, not a new model architecture or a separate teacher model. Its core contribution is to combine audio-grounded evidence validation, selective error rescue, directional OPSD, and answer-verified SAPO-based policy optimization into a deployment-compatible training pipeline.

## Related work

- [FireRedAudio](https://arxiv.org/pdf/2608.24168): Thanks to the FireRed Team for providing a strong audio-understanding base model.
- [SWIFT](https://arxiv.org/pdf/2408.05517): Thanks to ms-swift for providing a concise and easy-to-use post-training framework.
