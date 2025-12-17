---
id: 3
title: Plan-Physical-AI-Textbook
stage: plan
date: 2025-12-16
surface: agent
model: gpt-4
feature: 001-physical-ai-textbook
branch: 001-physical-ai-textbook
user: UCHIHA ITACHI
command: /sp.plan
labels: ["planning", "education", "ai", "robotics", "textbook", "docusaurus", "fastapi"]
links:
  spec: ../specs/001-physical-ai-textbook/spec.md
  plan: ../specs/001-physical-ai-textbook/plan.md
  adr: null
  pr: null
files:
 - specs/001-physical-ai-textbook/plan.md
 - specs/001-physical-ai-textbook/research.md
 - specs/001-physical-ai-textbook/data-model.md
 - specs/001-physical-ai-textbook/quickstart.md
 - specs/001-physical-ai-textbook/contracts/textbook-api-contract.md
 - .specify/memory/constitution.md
 - QWEN.md
tests:
 - none
---

## Prompt

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType qwen`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.

## Response snapshot

Successfully completed the implementation planning for the Physical AI & Humanoid Robotics textbook project. The following artifacts were created:

**Implementation Plan**: 
- Path: `C:\Users\UCHIHA ITACHI\Desktop\qwen_book\specs\001-physical-ai-textbook\plan.md`
- Contains technical context, constitution check, project structure, and complexity tracking

**Research Document**:
- Path: `C:\Users\UCHIHA ITACHI\Desktop\qwen_book\specs\001-physical-ai-textbook\research.md`
- Contains technology decisions with rationale and alternatives

**Data Model**:
- Path: `C:\Users\UCHIHA ITACHI\Desktop\qwen_book\specs\001-physical-ai-textbook\data-model.md`
- Defines entities for textbook content, student interactions, and RAG functionality

**API Contracts**:
- Path: `C:\Users\UCHIHA ITACHI\Desktop\qwen_book\specs\001-physical-ai-textbook\contracts\textbook-api-contract.md`
- Specifies API endpoints for textbook access and chatbot functionality

**Quickstart Guide**:
- Path: `C:\Users\UCHIHA ITACHI\Desktop\qwen_book\specs\001-physical-ai-textbook\quickstart.md`
- Provides setup and contribution instructions

**Agent Context Updated**:
- Updated `QWEN.md` with project-specific information

The plan follows all constitutional principles and is ready for the tasks phase.

## Outcome

- ✅ Impact: Created comprehensive planning artifacts for textbook project
- 🧪 Tests: None required for planning phase
- 📁 Files: Created 5 planning documents and updated agent context
- 🔁 Next prompts: Ready for /sp.tasks to define implementation tasks
- 🧠 Reflection: Planning phase completed with all constitutional requirements met

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Define implementation tasks