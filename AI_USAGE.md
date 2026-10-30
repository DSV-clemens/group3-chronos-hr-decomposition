# AI Usage Declaration

This research project was completed by **Group 3** with the support of AI
tools. In accordance with
[DSV's AI policy](https://www.su.se/english/divisions/department-of-computer-and-systems-sciences/education/during-your-studies/dsvs-ai-policy),
we declare this use transparently below: which tools we used, what they did,
and how we integrated and verified their output. The short form of this
declaration appears in Section VII of the accompanying paper.

**AI tools used:**

- **Vibe (Mistral AI):** agentic AI assistant with a sandboxed execution
  environment (primary tool)
- **ChatGPT (OpenAI)** and **Claude (Anthropic):** conversational assistance
- **Grammarly:** wording and language feedback

## How the AI tools were employed

| Area | What the AI did | How we integrated and verified it |
| --- | --- | --- |
| Code development | Helped with syntax, debugging, and clear comments across all pipeline scripts (data preparation, calibration, synthesis, forecasting arms, analysis, WindowSHAP) | We reviewed, adapted, and committed all code; every number in the paper can be re-derived from the committed artifacts in this repository (see `docs/metrics_map.md`) |
| Viewer scripts | The scripts that generate the inspection viewers were written entirely by AI | We validated them functionally against the committed data and results |
| Synthetic data generation | Helped adapt the synthesis script to more precise physiological values | This part received particular scrutiny: parameters were checked against the MMASH calibration, and all reported numbers remain traceable to the committed manifests (DGP hash, parameter SHA-256, seeds) |
| Understanding concepts | Explained difficult concepts, broke down mathematical formulas, and explained statistical methods (STL decomposition, quantile forecasting, bootstrap confidence intervals, Shapley values) | We discussed and confirmed our understanding before implementing |
| Git and GitHub | Taught us how to communicate with the repository directly from the console | We consolidated this knowledge outside the project and applied it ourselves: all commits, pull requests, merges, and tags were performed by the group |
| Manuscript support | Compared and condensed earlier paper drafts, checked citation consistency and text–table agreement, and resolved LaTeX/IEEEtran formatting issues (float placement, hyperlinks, page budget) | We reviewed, corrected, and approved every change; all interpretations, conclusions, and the final wording are our own |
| Language | Grammarly suggested wording and grammar edits | We reviewed and applied changes selectively |

## Work performed by the group itself

- Comparing and synthesizing our three preparatory literature reviews and
  developing the research question
- Designing the experimental pipeline and executing it locally
- All repository operations: commits, pull requests, merges, and tags
- Review, correction, and approval of every AI-supported draft before it was
  committed

## Examples of the questions we asked

- "Can you explain STL decomposition with a simple example?"
  (understanding the method)
- "Why does Chronos in the raw arm sometimes forecast a flat line? Give me a
  few theories we could test." (investigating model behavior)
- "How do I adjust the .tex file so it doesn't stick out past the margin?"
  (LaTeX formatting)
- "What critical questions could one ask about this document?" (finding
  logical gaps and preparing for the examination discussion)
- "I no longer know what I programmed here, but it works. Can you walk me
  through it and add comments?" (documenting code from a late-night session)

## Responsibility

Per DSV's AI policy, the group bears complete responsibility for the
originality and content of this submission. All AI-generated material was
reviewed, adapted where needed, and verified by the group members before
submission. The final submission is the group's own work product under the
course's examination rules.
