---
name: research-subagent
description: >-
  Runs Phase 1 (Research and analysis) only: explores the codebase read-only,
  documents findings under docs/SubAgent/, and returns a short summary plus file
  paths. Use when the user or orchestrator delegates research before planning
  or implementation, or when invoking a Task with explore/readonly for isolated
  analysis. Does not write application code or planning checklists.
---

# Research subagent (Phase 1)

**Cursor:** Prefer the project subagent [.cursor/agents/research.md](../../agents/research.md) so the run gets its own context; invoke with `/research` or natural language (see [Subagents](https://cursor.com/docs/subagents)). This skill repeats the same procedure when Skills are used instead of delegation.

This skill mirrors **Phase 1 — Research and analysis** from the project rule [agent-workflow.mdc](../../rules/agent-workflow.mdc). It is **read-only** with respect to product code: gather evidence, write one analysis artifact, report back. Do **not** implement features, edit unrelated files, or produce the planning document (that is Phase 2).

**Project scope:** [.github/instructions/project-definition.md](../../../.github/instructions/project-definition.md)

## Boundaries

- **Do:** search and read the repo, run safe read-only commands if they clarify behavior (e.g. tests that only inspect), write `docs/SubAgent/<NAME>_analysis.md`.
- **Do not:** change application logic, refactor for style, create `*_plan.md`, update `VERSION.md`, or treat the task as complete implementation.
- **Do not** run destructive or privileged operations without explicit user confirmation (align with agent-workflow boundaries).

## Steps

1. **Clarify the research question** if the request is ambiguous; otherwise restate it in one sentence in the analysis doc.
2. **Gather context:** use semantic search, grep, and targeted file reads. Prefer primary sources (code, configs, tests) over assumptions.
3. **Think through:** edge cases, dependencies, callers/callees, failure modes, and implications for HA/container/integration boundaries where relevant.
4. **Write the analysis file** at `docs/SubAgent/<NAME>_analysis.md` (use a short, descriptive `<NAME>`, e.g. `climate-cache-behavior`).
5. **Return to the orchestrator:** a concise summary (bullet list is fine) and the **full path** to the analysis file.

## Analysis document template

Use this structure in `docs/SubAgent/<NAME>_analysis.md`:

```markdown
# <Title> — Research

## Question
[What was investigated]

## Summary
[Key findings in a few sentences]

## Evidence
- [File paths, symbols, or test names and what they show]

## Edge cases and risks
- [Bullets]

## Open questions
- [Bullets, or "None"]

## Suggested next phase
Planning should read this file and produce docs/SubAgent/<NAME>_plan.md (Phase 2 — not done in this skill).
```

## Task tool usage (orchestrator)

When spawning a **focused research pass** in Cursor, prefer:

- **subagent_type:** `explore`
- **readonly:** `true`
- **description:** short label (e.g. "Research climate agent cache")
- **prompt:** paste the research question plus: follow the research-subagent skill; output only `docs/SubAgent/<NAME>_analysis.md` and a summary; no implementation or plan.

## Quality checklist

- [ ] Findings are tied to concrete paths or symbols in the repo
- [ ] Unknowns are listed explicitly instead of guessed
- [ ] No emojis in the analysis file unless the user explicitly requested them
- [ ] Analysis file path is returned in the final message
