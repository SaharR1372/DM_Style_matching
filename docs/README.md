# Documentation

| document | what it covers |
| --- | --- |
| [method.md](method.md) | the method: L_MMD, the Style Matching module (L_MM, L_CM) and the Intra-Class Diversity module (L_ICD), and why the released form of each is what it is |
| [coreset.md](coreset.md) | the eight coreset selection baselines, what each optimises, when it works and when it does not |
| [configs.md](configs.md) | the YAML schema: every key, its default and its meaning; inheritance and `--set` overrides |
| [datasets.md](datasets.md) | where each dataset comes from, how to prepare TinyImageNet and ImageNet, how to add one |
| [results.md](results.md) | the measured numbers, the protocol behind each, and what they say |
| [extending.md](extending.md) | adding a loss term, a selector, a dataset or an architecture |

## Where to start

- **Running the released method:** the main [README](../README.md), then
  [configs.md](configs.md) when you want to change something.
- **Comparing against coreset baselines:** [coreset.md](coreset.md).
- **Understanding or citing the method:** [method.md](method.md), then
  [results.md](results.md).
- **Building on it:** [extending.md](extending.md).
