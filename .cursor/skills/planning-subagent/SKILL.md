---
name: planning-subagent
description: >-
  Runs Phase 2 (Planning) only: reads an existing research analysis under
  docs/SubAgent/, produces a step-by-step plan and checklist as
  docs/SubAgent/<NAME>_plan.md, and returns a summary plus path. Use when the
  user or orchestrator has completed Phase 1 and needs an approved plan before
  implementation. Does not implement product code or replace user plan approval.
---

# Planning subagent (Phase 2)

**Cursor:** Prefer the project subagent [.cursor/agents/planning.md](../../agents/planning.md) for a separate context; invoke with `/planning` or natural language (see [Subagents](https://cursor.com/docs/subagents)). This skill repeats the same procedure when Skills are used instead of delegation.

This skill mirrors **Phase 2 — Planning** from the project rule [agent-workflow.mdc](../../rules/agent-workflow.mdc). It turns analysis into an actionable, ordered plan. **Phase 1** is the [research-subagent](../research-subagent/SKILL.md) skill; **Phase 3** is implementation after explicit user approval of the plan.

**Project scope:** [.github/instructions/project-definition.md](../../../.github/instructions/project-definition.md)

## Boundaries

- **Do:** read `docs/SubAgent/<NAME>_analysis.md` (path may be given explicitly if the name differs), optionally re-read cited code to sanity-check the plan, write **only** `docs/SubAgent/<NAME>_plan.md` (same `<NAME>` as the analysis unless the user specifies otherwise).
- **Do not:** implement features, edit application or test code, drive large refactors, or skip listing risks and verification steps.
- **Do not** treat the work as approved: the orchestrator must present the plan in chat (or Plan mode) and obtain **explicit user approval** before Phase 3.
- **Do not** run destructive or privileged commands without explicit user confirmation.

## Prerequisites

- A completed analysis file from Phase 1, typically `docs/SubAgent/<NAME>_analysis.md`.
- If that file is missing, either stop and ask for the path to the analysis or for Phase 1 to be run first; do not invent analysis from scratch without a clear charter.

## Steps

1. **Read the analysis** end-to-end; note open questions and evidence links.
2. **Resolve scope:** align the plan with the research question, project boundaries (HA container vs integration), and any user constraints stated in chat.
3. **Think through:** ordering constraints, edge cases from the analysis, rollback or safety notes, and what would invalidate the plan.
4. **Write the plan file** at `docs/SubAgent/<NAME>_plan.md`.
5. **Return to the orchestrator:** short summary (what the plan delivers, major steps, top risks) and the **full path** to the plan file. Remind that implementation must wait for user approval.

## Plan document template

Use this structure in `docs/SubAgent/<NAME>_plan.md`:

```markdown
# <Title> — Plan

## References
- Analysis: docs/SubAgent/<NAME>_analysis.md
- [Other links if needed]

## Goal
[What success looks like]

## Non-goals
- [What is explicitly out of scope]

## Assumptions
- [Bullets; flag uncertain assumptions]

## Implementation checklist
Ordered steps for Phase 3 (check off during implementation):

- [ ] Step 1 — ...
- [ ] Step 2 — ...
- [ ] ...

## Files and areas (expected)
- [Paths or modules likely touched; optional if unknown]

## Testing and verification
- [Commands, manual checks, or tests to run after changes]

## Risks and mitigations
- [Bullets]

## Versioning
- [If the change is a user-visible feature or notable behavior change, note that [VERSION.md](../../../VERSION.md) and changelog should be updated per project semver rules; otherwise "Not applicable" or "Patch only — see VERSION.md rule"]

## Approval gate
Do not start Phase 3 until the user explicitly approves this plan (or an updated revision).
```

## Task tool usage (orchestrator)

Planning must **create or update** a markdown file under `docs/SubAgent/`. Prefer:

- **subagent_type:** `generalPurpose`
- **description:** short label (e.g. "Plan climate cache follow-up")
- **prompt:** include the path to `*_analysis.md`, the desired `<NAME>` for `*_plan.md`, and: follow the planning-subagent skill; write only the plan file; no application code changes; remind orchestrator to seek user approval before implementation.

If the environment must stay read-only, return the full plan in the message and have the orchestrator save it to `docs/SubAgent/<NAME>_plan.md` manually.

## Quality checklist

- [ ] Every major step from the analysis maps to a checklist item or is deferred with reason
- [ ] Ordering reflects dependencies (migrations before callers, tests aligned with changes, etc.)
- [ ] Open questions from the analysis are addressed or carried forward explicitly
- [ ] No emojis in the plan file unless the user explicitly requested them
- [ ] Plan file path is returned in the final message
