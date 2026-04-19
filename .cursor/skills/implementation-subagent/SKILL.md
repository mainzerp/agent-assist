---
name: implementation-subagent
description: >-
  Runs Phase 3 (Implementation) only: executes an already user-approved plan at
  docs/SubAgent/<name>_plan.md, verifies per the plan, updates VERSION.md when
  SemVer rules apply. Use when Skills are used instead of the implementation
  subagent in .cursor/agents/. Does not perform Phase 1 or 2.
---

# Implementation subagent (Phase 3)

**Cursor:** Prefer the project subagent [.cursor/agents/implementation.md](../../agents/implementation.md) for a separate context; invoke with `/implementation` or natural language (see [Subagents](https://cursor.com/docs/subagents)). This skill repeats the same procedure when Skills are used instead of delegation.

This skill mirrors **Phase 3 — Implementation** from [agent-workflow.mdc](../../rules/agent-workflow.mdc).

**Project scope:** [.github/instructions/project-definition.md](../../../.github/instructions/project-definition.md)

**Versioning:** [.cursor/rules/version-semver.mdc](../../rules/version-semver.mdc)

## Prerequisites

- `docs/SubAgent/<name>_plan.md` exists.
- The **user has explicitly approved** that plan in chat (or equivalent). If not confirmed, stop and send the parent back to the approval gate.

## Steps

1. Read the plan file completely; follow its checklist order.
2. Implement without re-opening settled design choices from the plan unless blocked; if blocked, report and stop.
3. Stay scoped: no unrelated refactors; match codebase conventions.
4. Run verification steps the plan specifies (tests, lint, manual checks).
5. Update `VERSION.md` only when the change warrants it per SemVer (features, notable fixes, etc.).

## Return to orchestrator

- Summary of edits (files + intent).
- Checklist status vs plan.
- Verification results.
- Remaining gaps or follow-ups.

No emojis unless the user explicitly requested them.

## Task tool usage (orchestrator)

- **subagent_type:** `generalPurpose` (typical for edits and tests).
- **prompt:** Include plan path, explicit line that the **user approved** the plan, and any constraints from chat.
