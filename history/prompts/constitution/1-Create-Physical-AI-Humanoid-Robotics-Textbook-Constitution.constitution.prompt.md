---
id: 1
title: Create-Physical-AI-Humanoid-Robotics-Textbook-Constitution
stage: constitution
date: 2025-12-16
surface: agent
model: gpt-4
feature: none
branch: master
user: UCHIHA ITACHI
command: /sp.constitution
labels: ["constitution", "education", "ai", "robotics"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

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


/sp.constitution  
You are an expert AI architect, technical author, and full-stack engineer.

Your task is to create a COMPLETE, UNIFIED BOOK PROJECT titled:

"Create a Textbook for Teaching Physical AI & Humanoid Robotics"

Use:
- Claude Code
- Spec-Kit Plus
- Docusaurus (for book UI & documentation)
- FastAPI (backend)
- Next.js (frontend integration if needed)
- Python
- OpenAI Agents / ChatKit SDK
- Qdrant Cloud (Free Tier)
- Neon Serverless Postgres
- GitHub Pages deployment

────────────────────────────────
CORE OBJECTIVES
────────────────────────────────

1. Write a FULL TEXTBOOK using **Docusaurus**
2. Structure content using **Spec-Kit Plus specifications**
3. Deploy the book to **GitHub Pages**
4. Build an **Integrated RAG Chatbot**
5. Embed chatbot inside the book UI
6. Chatbot must:
   - Answer questions from the whole book
   - Answer questions ONLY from user-selected text
   - Use vector search (Qdrant)
   - Store metadata in Neon Postgres
7. Follow clean architecture & production-ready folder structure

────────────────────────────────
COURSE THEME
────────────────────────────────

Physical AI & Humanoid Robotics  
Focus: Embodied Intelligence  
Goal: Bridge the digital brain (AI) with the physical body (robots)

Students should:
- Apply AI knowledge to humanoid robots
- Work in simulation and real environments
- Learn perception, planning, navigation, and manipulation

────────────────────────────────
BOOK STRUCTURE (CHAPTERS)
────────────────────────────────

Create chapters STRICTLY based on the following modules:

### Chapter 1: Introduction to Physical AI & Embodied Intelligence
- What is Physical AI
- Digital Brain vs Physical Body
- Role of Humanoid Robots
- Course roadmap & outcomes

────────────────────────────────

### Chapter 2: The Robotic Nervous System (ROS 2)
(Module 1)

Include:
- ROS 2 architecture
- Nodes, Topics, Services
- ROS 2 vs ROS 1
- rclpy: Bridging Python AI agents to robot controllers
- URDF (Unified Robot Description Format)
- Humanoid robot modeling
- Diagrams & code snippets

────────────────────────────────

### Chapter 3: The Digital Twin
(Module 2)

Include:
- Concept of Digital Twins
- Gazebo physics simulation
- Gravity, collisions, joints
- Sensor simulation:
  - LiDAR
  - Depth Cameras
  - IMU
- Unity for:
  - High-fidelity rendering
  - Human-robot interaction
- Comparison: Gazebo vs Unity

────────────────────────────────

### Chapter 4: The AI-Robot Brain (NVIDIA Isaac™)
(Module 3)

Include:
- NVIDIA Isaac Sim overview
- Photorealistic simulation
- Synthetic data generation
- Isaac ROS
- Hardware-accelerated VSLAM
- Nav2 for humanoid navigation
- Path planning for bipedal robots

────────────────────────────────

### Chapter 5: Vision-Language-Action (VLA)
(Module 4)

Include:
- What is VLA
- LLMs + Robotics
- Voice-to-Action pipeline
- OpenAI Whisper integration
- Natural language → ROS 2 actions
- Cognitive planning
- Examples:
  - "Clean the room"
  - "Pick the bottle"

────────────────────────────────

### Chapter 6: Capstone Project – Autonomous Humanoid
(FINAL PROJECT)

Robot must:
- Receive voice command
- Convert speech → text
- Plan actions using LLM
- Navigate environment
- Avoid obstacles
- Identify object via computer vision
- Manipulate object

Provide:
- Architecture diagram
- Step-by-step implementation
- Simulation setup
- Evaluation metrics

────────────────────────────────
Docusaurus UI REQUIREMENTS
────────────────────────────────

- Modern, clean, professional UI
- Dark + Light mode
- Sidebar navigation by modules
- Code blocks with syntax highlighting
- Diagrams & images placeholders
- Embedded chatbot UI panel
- Mobile responsive

────────────────────────────────
RAG CHATBOT ARCHITECTURE
────────────────────────────────

Backend:
- FastAPI
- OpenAI Agents / ChatKit SDK
- Qdrant Cloud (vector storage)
- Neon Serverless Postgres (metadata)
- Text chunking & embeddings
- Selection-based answering (highlighted text only)

Frontend:
- Embedded in Docusaurus
- Chat UI component
- Ask about:
  - Whole book
  - Selected text only

────────────────────────────────
PROJECT FOLDER STRUCTURE
────────────────────────────────

Create a clean monorepo:

/physical-ai-textbook
│
├── frontend/
│   ├── docusaurus/
│   │   ├── docs/
│   │   ├── src/
│   │   ├── static/
│   │   └── docusaurus.config.js
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── rag/
│   │   ├── embeddings/
│   │   ├── database/
│   │   └── main.py
│
├── spec/
│   └── book.spec.yaml
│
├── scripts/
│   ├── ingest_docs.py
│   └── deploy.sh
│
└── README.md

────────────────────────────────
SPEC-KIT PLUS
────────────────────────────────

- Generate `book.spec.yaml`
- Define:
  - Mission
  - Goals
  - Audience
  - Success metrics
  - Reusability for RAG
- Ensure specs drive content generation

────────────────────────────────
OUTPUT REQUIREMENTS
────────────────────────────────

1. Complete book content (Markdown)
2. Docusaurus setup
3. FastAPI RAG backend
4. Chatbot embedded UI
5. Deployment instructions
6. Production-ready code
7. Clean documentation

────────────────────────────────
IMPORTANT RULES
────────────────────────────────

- Be concise but complete
- Use real-world examples
- Follow best practices
- Make content reusable for RAG
- Assume students have basic Python & AI knowledge

START EXECUTION NOW.

## Response snapshot

The constitution for the Physical AI & Humanoid Robotics Textbook project has been successfully created. The document establishes six core principles tailored to the project requirements, along with sections for technology stack requirements, development workflow, and governance.

New Version: 1.0.0 (Initial version for the Physical AI & Humanoid Robotics Textbook project)

Bump Rationale: This is an initial version creation for the specific textbook project focusing on Physical AI & Humanoid Robotics, establishing the foundational principles for content creation, technology integration, and educational approach.

Files Flagged for Manual Follow-up:
- `.specify/templates/plan-template.md` - The "Constitution Check" section references the constitution file and may need fine-tuning to ensure it properly validates against the new principles.

Suggested Commit Message: `docs: establish constitution for Physical AI & Humanoid Robotics textbook v1.0.0`

The six core principles established are:
1. Modular Content Architecture
2. Technology Integration First
3. Test-Driven Learning (NON-NEGOTIABLE)
4. Simulation-First Approach
5. Embodied Intelligence Focus
6. Production-Ready Education

## Outcome

- ✅ Impact: Created foundational constitution for textbook project
- 🧪 Tests: None required for this documentation task
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Can proceed with creating textbook content and architecture
- 🧠 Reflection: The project now has a solid foundation with defined principles and guidelines

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating the textbook content based on the constitution