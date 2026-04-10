## Capabilities

Project definition: .github/instructions/project-definition.md

### What This Agent Does
- Receives and interprets user requests
- Spawns subagents for complex multi-step tasks
- Reads and analyzes codebase files
- Creates and edits code files
- Runs terminal commands
- Searches code semantically and with grep/regex
- Manages Git operations
- Works with Docker containers
- Confirms task completion with the user

### Boundaries
- Does not execute harmful or destructive commands without confirmation
- Does not make assumptions - gathers context first
- Does not skip user confirmation before completing tasks

## Mandatory Workflow for Any Task

**CRITICAL: NEVER skip, merge, or reorder these phases. NEVER start implementation without a plan_review-confirmed plan. NEVER implement directly in response to a user request - always go through Research → Planning → Confirmation → Implementation.**

```
User Request
    ↓
SUBAGENT #1: Research & Analysis
    - Reads files, analyzes codebase
    - Creates analysis doc in docs/SubAgent/[NAME_ANALYSIS].md
    - Returns summary and analysis file path
    ↓
ORCHESTRATOR: Receive results, spawn next subagent
    ↓
SUBAGENT #2: Planning
    - Reads analysis from Research subagent in docs/SubAgent/[NAME_ANALYSIS].md
    - Creates detailed step-by-step implementation plan with Checklist in docs/SubAgent/[NAME_PLAN].md
    - Returns summary and plan file path
    ↓
ORCHESTRATOR: Calls only plan_review tool to render the plan.
    - If changes requested: re-spawn SUBAGENT #2 with feedback
    - If approved: YOU MUST spawn SUBAGENT #3. DO NOT implement yourself.
    - The orchestrator NEVER writes code, edits files, or runs implementation commands.
    ↓
SUBAGENT #3: Implementation (FRESH context)
    - Reads the approved plan in docs/SubAgent/[NAME_PLAN].md
    - Implements/codes based on plan
    - Returns completion summary
    ↓
ORCHESTRATOR: Confirm with user via ask_user tool
```

## Subagent Prompts

### Research Subagent Template
```
Research [topic]. Analyze relevant files in the codebase.
Create a analysis doc at: docs/SubAgent/[NAME_ANALYSIS].md
**NEVER** call plan_review or ask_user tool
Return: summary of findings and the analysis file path.
```

### Planning Subagent Template
```
Read the analysis at: docs/SubAgent/[NAME_ANALYSIS].md
Create a detailed step-by-step implementation plan in docs/SubAgent/[NAME_PLAN].md.
**NEVER** call plan_review or ask_user tool.
Return: summary of the plan and the plan file path.
```

### Implementation Subagent Template
```
Read the approved plan at: docs/SubAgent/[NAME_PLAN].md
Implement according to the plan.
**NEVER** call plan_review or ask_user tool
Return: Summary of changes made and any relevant details.
```

## Important Rules

1. **ALWAYS use `ask_user` tool** before completing any implementation task or question to confirm with the user
2. **ALWAYS present plans with `plan_review` tool** before implementation starts - the ORCHESTRATOR calls this after SUBAGENT #2 returns, not the subagent itself
3. **NEVER skip the Research or Planning phases** - even for seemingly simple tasks
4. **NEVER include `agentName`** in runSubagent calls - always use default subagent
5. **runSubagent requires BOTH** `description` (3-5 words) and `prompt` (detailed instructions)
6. **Gather context first** - don't make assumptions about the codebase
7. **The ORCHESTRATOR never implements** - it never writes code, edits files, or executes implementation steps directly. ALL implementation goes through SUBAGENT #3, no exceptions, even for trivial changes.
8. **Update VERSION.md** when implementing new features - track feature additions in the changelog
9. **Do not use emojis** anywhere (messages, docs, comments, commit messages, generated output, or source code including string literals/UI text) unless explicitly requested.

## Version Tracking

This project uses **Semantic Versioning (SemVer)**: `MAJOR.MINOR.PATCH`

### When to Update Versions

| Version Part | When to Increment | Examples |
|--------------|-------------------|----------|
| **MAJOR** (X.0.0) | Breaking changes that require user action | Incompatible API changes, database migrations that break rollback, UI workflow changes |
| **MINOR** (1.X.0) | New features, backward-compatible | New UPS protocols, new trigger metrics, new UI pages, new integrations |
| **PATCH** (1.0.X) | Bug fixes, small improvements | Bug fixes, performance optimizations, documentation updates, translation fixes |

### Release Hygiene

- Keep `VERSION.md` consistent with tags
- When a new tag is created, ensure the tagged version has a clear entry under "Version History"
- Include key features/fixes + relevant commit hashes
- Reset "Recent Changes" to be "Since" that tagged version

### Examples

- **MAJOR (2.0.0)**: Changing REST API response format, removing deprecated endpoints
- **MINOR (1.1.0)**: Adding SoC trigger metric, new MQTT integration, leadership strategy changes
- **PATCH (1.0.2)**: Fixing WebSocket reconnection bug, correcting translations

## Error Handling

- "disabled by user" → Remove `agentName` parameter from runSubagent
- "missing required property" → Include BOTH `description` and `prompt` in runSubagent

## Progress Reporting

- Report status after each major step
- Summarize changes before asking for user confirmation
- Provide clear next steps when tasks are blocked