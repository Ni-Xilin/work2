# AGENTS.md

## Scope

This file applies to the current directory and all subdirectories beneath it.

## Project Context

This workspace contains an Augur-related research project focused on traffic correlation attack defense. The main implementation lives under `Generator_Trainer/`, and project documents should be written under `work2/docs/` unless the user explicitly asks for a different location.

## Working Style

- Preserve the Augur research context unless the user explicitly changes it.
- When proposing or implementing a second work point, prefer continuity with the existing Augur task, datasets, and evaluation targets.
- Keep explanations rigorous, readable, and beginner-friendly.
- Avoid overly fragmented writing with too many tiny nested bullet points.
- Prefer a balanced document style: short sections, summary tables, diagrams, code blocks, and explanatory prose.

## Documentation Rules

- Write new project documents under `work2/docs/`.
- When a document is concept-heavy, include visual aids where helpful, such as:
  - Mermaid flowcharts
  - comparison tables
  - pseudocode blocks
  - symbol or terminology tables
- Do not use dense wall-of-text writing when diagrams or tables would improve readability.
- When discussing reprogramming, distinguish clearly between:
  - non-text modalities that must be mapped into an LLM-compatible semantic space
  - prompt text that already lives in the language space and acts as a semantic anchor

## Documentation Writing Contract

- Treat technical writing as structured reasoning rather than feature listing. Prefer a first-principles progression: identify the underlying representation bottleneck, physical constraint, or engineering limitation first; then introduce the required mechanism; then explain what capability or stability improvement follows from that mechanism.
- Keep the page readable. Do not write in either extreme:
  - no dense wall-of-text blocks that are visually tiring
  - no skeletal outlines where each point contains only a few words
- Bullets are allowed and often preferred, but each bullet must be a self-contained explanation. A good default is: one lead sentence, followed by `2-4` medium-length bullets, where each bullet explains a complete idea rather than a label.
- Each major point should explain three layers whenever possible:
  - what the mechanism is at the operational level
  - why it is necessary under the task, resource, or modeling constraints
  - how it affects stability, interpretability, deployability, or experimental fairness
- Use explicit causal transitions between sections. Prefer bridges such as:
  - because the first work point exposes a representation bottleneck, the second work point must introduce a new mapping mechanism
  - based on the semantic mapping above, the next challenge is how to keep the generated output physically deployable
  - to align the mathematical objective with engineering implementation, the training loop must be reorganized as follows
- For concept-heavy sections, prefer this composition instead of one long paragraph:
  - a short lead sentence explaining what question the section answers
  - one or more substantial explanatory bullets
  - a table, Mermaid diagram, formula block, or pseudocode block when it reduces cognitive load
- Use visual aids proactively. When the content involves data flow, module hierarchy, decision paths, training loops, or state transitions, add Mermaid diagrams that complement the prose rather than restating it mechanically.
- Use tables to reduce reading pressure whenever prose becomes hard to scan. Tables are especially encouraged for:
  - symbol explanations
  - module responsibilities
  - dataset and metric summaries
  - experiment group definitions
  - ablation settings
  - model selection tradeoffs
- When formulas are needed in Markdown documents, prefer block LaTeX with `$$ ... $$`. Formulas must constrain the logic rather than decorate the page, and the surrounding text should explain variable meanings and the role of the equation.
- When code or pseudocode appears inside project documents, all comments, docstrings, parameter explanations, return-value explanations, and error messages should be written in Chinese unless the user explicitly requests English.
- For long technical documents, add navigation aids near the top when helpful, such as:
  - a quick navigation section
  - a short reading-path note
  - a summary card for key choices like `3B / 7B / 14B`
- Improve readability with restrained visual emphasis. Appropriate tools include:
  - emoji in major headings
  - callout blocks for labels such as `Default Plan`, `Risk Note`, `Key Conclusion`, or `Final Recommendation`
  - light color highlighting for key judgment sentences when Markdown rendering supports it
  - sections such as `Common Misunderstandings vs Correct Reading`, `Terminology Quick Reference`, or `Model Selection Decision Graph`
- Visual emphasis must remain selective. Do not colorize or decorate everything. Highlight only conclusion sentences, risk boundaries, default recommendations, or easily misunderstood claims.
- Maintain a cold, precise, non-decorative tone in technical documents. Avoid inflated adjectives. The document should feel strong because its logic is constrained, not because its wording is promotional.

## Code and Experiment Guidance

- Reuse the existing datasets and task setup from `Generator_Trainer/` unless the user explicitly requests new data.
- Keep comparisons fair with the first work point: same datasets, same target models, same evaluation protocol when possible.
- Prefer lightweight architectural additions before introducing heavy new modules.
- Do not claim a module performs semantic alignment unless the mechanism actually supports that claim.

## Editing Discipline

- Make the smallest change that cleanly satisfies the current request.
- Preserve user intent over stylistic preference.
- If a file becomes hard to read, restructure it instead of merely appending more text.

## Practical Default

If the user asks for planning, explanation, or document writing in this workspace, default output location is:

`work2/docs/`
