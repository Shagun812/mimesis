# MIMESIS
### Causal Localization of Prompted Sandbagging in Language Models

MIMESIS investigates the internal mechanisms underlying **prompted sandbagging**: a model's ability to underperform on a task it can otherwise solve when explicitly instructed to do so.

Using activation patching on Qwen3-4B, this project examines where the internal difference between honest and sandbagging behavior becomes causally accessible, whether correct answers can be restored through targeted interventions, and how specific these effects are to the answer being expressed.

## Research Question

**Where in a language model does prompted sandbagging become causally recoverable, and can intervening on its internal representations restore correct behavior?**

The central objective is to move beyond behavioral observation and investigate the causal role of internal model states in producing suppressed performance.

## Experimental Setup

| Component | Description |
|---|---|
| Model | Qwen3-4B |
| Evaluation task | WMDP-Bio |
| Behavioral conditions | Honest, Sandbagging, Counterfactual |
| Transformer layers | 36 |
| Working evaluation set | 56 questions |
| Primary intervention | Layer-wise residual-stream activation patching |
| Total interventions | 2,016 |

Three behavioral conditions were used:

- **Honest:** The model answers questions normally.
- **Sandbagging:** The model is instructed to answer incorrectly on questions associated with hazardous biological knowledge.
- **Counterfactual:** The hazardous framing is retained, but the instruction to underperform is removed.

A calibration experiment was used to validate the behavioral setup. The working population consisted of 56 questions answered correctly in the Honest condition but incorrectly in the Sandbagging condition.

For each question, the residual stream at the final input position was replaced with the corresponding activation from the alternative behavioral condition. This intervention was repeated layer by layer to measure how the causal effect of the injected state changes across network depth.

## Key Findings

### 1. A sharp transition in causal recovery

Patching the Honest residual state into the Sandbagging trajectory produced a marked increase in correct-answer recovery around Layer 24.

| Layer | Correct-answer recovery |
|---|---:|
| L18 | 41.1% |
| L21-L22 | 66.1% |
| L23 | 67.9% |
| L24-L25 | 91.1% |
| L26-L29 | 92.9% |
| L30 | 94.6% |
| L31-L32 | 98.2% |
| L34-L35 | 100% |

Recovery increased by **23.2 percentage points between L23 and L24**. A bootstrap analysis using 5,000 resamples gave a 95% interval of 12.5 to 33.9 percentage points for this increase.

This identifies a sharp population-level transition in the causal recoverability of Honest behavior. It does not, by itself, establish where sandbagging originates.

### 2. Recovery depends strongly on answer identity

To test whether recovery was simply caused by injecting an arbitrary Honest activation, L24 activations from other questions were used as donors.

- **Same correct answer:** 53 of 55 questions recovered, or 96.4%.
- **Different correct answer:** 1 of 55 questions recovered, or 1.8%.

The large difference suggests that the effect of the injected state is strongly dependent on the answer associated with the donor question, rather than being a generic consequence of replacing the activation with an Honest state.

### 3. The effect is bidirectional

The intervention was also reversed by injecting Sandbagging L24 activations into Honest trajectories.

Of the 56 questions answered correctly in the Honest baseline, **45 became incorrect** after the intervention. Only 11 remained correct.

This provides complementary causal evidence that the L24 residual state participates in separating Honest and Sandbagging behavior. Honest-state interventions can restore correct answers, while Sandbagging-state interventions can disrupt them.

## Interpretation

Taken together, the experiments suggest that the Honest/Sandbagging behavioral difference becomes strongly causally accessible around Layer 24, with effects that depend heavily on the answer being expressed.

One hypothesis is that middle-to-late computation integrates instruction context with an already-developed answer representation, influencing how that information is expressed. However, the current experiments cannot distinguish this from an alternative in which the relevant answer information or behavioral difference develops later.

**Layer 24 is therefore interpreted as a major point of causal separation between the two behavioral states, not as an established origin of the sandbagging mechanism.**

## Limitations

- The investigation uses one model, Qwen3-4B, and one evaluation task, WMDP-Bio.
- The primary working population contains 56 questions selected because they flip from correct to incorrect between the Honest and Sandbagging conditions.
- The intervention replaces the full residual-stream activation at a single token position, so the results do not isolate individual features or circuits.
- The layer-wise recovery transition does not establish when correct-answer information first becomes represented or when the sandbagging instruction first affects computation.
- The results require replication across models, tasks, prompts, and intervention methods before broader generalization is warranted.

## Future Directions

Potential follow-up experiments include:

- **Layer-wise answer probing:** Test when correct-answer information becomes decodable in the residual stream.
- **Answer-specific activation analysis:** Investigate how activation differences relate to the answer being expressed.
- **Per-question recovery analysis:** Identify factors that explain why some questions become causally recoverable earlier than others.
- **Cross-model replication:** Test whether similar causal transitions occur in other models and sandbagging setups.

## Full Research Write-up

The complete research write-up provides the detailed methodology, experimental results, intervention controls, analysis, and discussion.

**[Read the full research write-up & Results](https://drive.google.com/drive/folders/1uTBtvD-SLWSryyx7WjQhjF1VhciO6tS1?usp=drive_link)**

## Status

Research experiments and analysis completed for the current investigation. Broader replication and further investigation of the origin of the observed causal transition remain open.

## Citation

If referencing this work, please use the project repository and the full research write-up linked above.