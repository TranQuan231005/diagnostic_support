# AI Agent Rules — diagnostic_support

> These rules govern ALL Agent behavior in this workspace.
> They are derived from the project's AI usage policy and extended with real-world business engineering best practices.
> Scope: **Workspace-only** (`diagnostic_support` project).

---

## 1. Six Core Principles (ALWAYS ACTIVE)

These principles apply to every single interaction, without exception.

1. **Understand before coding** — Before writing any code, the Agent must fully understand the problem, the existing codebase context, and the expected behavior. Never generate code speculatively.
2. **Plan first** — For any task touching more than 1 file OR involving an architectural decision, the Agent MUST produce a written implementation plan and obtain explicit user approval before executing. Use Planning Mode.
3. **Small prompts, small changes** — Implement one module at a time, one change at a time. Never batch unrelated changes into a single step.
4. **Run & verify every change** — After every code change, the Agent must run the relevant tests or validation commands and report the results before moving on.
5. **AI output is untrusted** — The Agent must never assert that a feature works without verified evidence (test output, logs, or user confirmation). Never claim correctness without proof.
6. **Explain the code** — For every non-trivial code block generated, the Agent must provide a brief explanation of what it does and why it was designed that way.

---

## 2. Standard Development Workflow

The Agent MUST follow this workflow in order. Steps cannot be skipped.

```
Define → Design → Break Down → Build MVP → Implement → Test → Refactor → Commit → Document
```

### 2.1 Define & Design
- **Rule**: Do NOT write code until the problem and system structure are clearly defined.
- **Define** (must capture): Project name, problem statement, target users, input/output spec, core features, technology choices, evaluation criteria.
- **Design** (must capture): System architecture, module breakdown, folder structure, data flow, interfaces/APIs, milestones, technical risks.

### 2.2 Build MVP → Implement Module by Module
- Build the MVP first before full implementation.
- Work on **one module per session**, **one change per commit**.

### 2.3 Debug & Test Systematically
- **Debug prompt structure the Agent must use**:
  - Expected output
  - Actual behavior
  - Error message
  - Relevant code snippet
  - Environment details
  - Recent changes
  - → Agent produces: root-cause analysis → ranked hypotheses → verification steps → minimal fix
- **Minimum test coverage per feature**:
  - Normal/happy-path cases
  - Invalid input cases
  - Edge cases
  - Failure/error cases
  - Integration tests
  - (For ML/AI modules): data split check, baseline comparison, metrics validation, leakage check, error analysis

---

## 3. Testing Mandate

> **MANDATORY**: The Agent MUST write or update at least one test for every new function or module it creates.

- Tests must be co-committed with the implementation — never deferred.
- The Agent must run the full test suite after every change and report pass/fail results.
- Deleting or disabling tests to make a pipeline pass is a **prohibited behavior** (see Section 6).

---

## 4. Planning Mode Enforcement

> **MANDATORY**: The Agent MUST enter Planning Mode (create `implementation_plan.md` and request user approval) before executing any task that:
> - Modifies more than 1 file, OR
> - Involves an architectural decision (new module, schema change, API change), OR
> - Is flagged as a breaking change.

The Agent must STOP and wait for explicit user approval before proceeding to execution.

---

## 5. Documentation Standards

### README.md (required for every project/module)
Must contain:
1. Problem overview & project statement
2. Main features
3. System architecture diagram or description
4. Folder structure
5. Installation / Run / Test instructions
6. Example input/output
7. Results & evaluation metrics
8. Limitations
9. Future improvements
10. Links to related files

### AI_Usage.md (required for every project)
Must contain:
- Tools used (name, version)
- Main purpose of each AI tool in the project
- Developer contribution vs. AI-generated contribution
- List of important AI-generated components with explanations

---

## 6. Prohibited Behaviors — HARD STOP

> When the Agent detects any of the following, it MUST **refuse to proceed** and require **explicit user approval** before continuing.

| # | Prohibited Behavior |
|---|---|
| 1 | Submitting or committing entirely AI-generated code without human review |
| 2 | Deleting or disabling tests to make a CI/CD pipeline pass |
| 3 | Including API keys, passwords, tokens, or sensitive data in prompts or code |
| 4 | Claiming a feature works when no verified test or evidence exists |
| 5 | Using code without running it or understanding it |
| 6 | Manipulating evaluation results to match expectations |
| 7 | Committing `.env`, tokens, or secrets to version control |
| 8 | Failing to disclose AI-generated contributions |
| 9 | Committing directly to `main` or `master` branch |
| 10 | Adding a dependency with a known CVE without flagging it |
| 11 | Modifying a public API or database schema without a breaking-change warning and explicit approval |
| 12 | Exposing PII or sensitive user data in any generated code, logs, or prompts |

---

## 7. Business Engineering Guardrails

### 7.1 Branching Strategy
- The Agent must NEVER commit directly to `main` or `master`.
- All changes must be proposed on a feature branch: `feature/<short-description>` or `fix/<short-description>`.
- The Agent must remind the user to create/switch to the correct branch before starting implementation.

### 7.2 Security-First
- Before generating any code that handles user data, authentication, or external services, the Agent MUST:
  1. Identify any PII or sensitive data in scope.
  2. Flag potential exposure risks.
  3. Propose mitigations (encryption, masking, env vars).
- **HARD STOP** if credentials or secrets are detected in code or prompts.

### 7.3 Code Review Gate
- Before proposing any pull request (PR) or completing a feature, the Agent MUST produce a self-review checklist:
  - [ ] Does the code meet the defined requirements?
  - [ ] Are all tests passing?
  - [ ] Are there no hardcoded secrets?
  - [ ] Is the code readable and explained?
  - [ ] Is documentation updated?
  - [ ] Are there no unintended side effects?
  - [ ] Are new dependencies vetted for security?

### 7.4 Observability
- Every new module or service the Agent creates MUST include:
  - Structured logging (INFO for normal flow, ERROR for exceptions).
  - Key operation timing/metrics where performance matters (e.g., inference latency, throughput).
  - Clear error messages with context (not bare exceptions).

### 7.5 Dependency Hygiene
- Before adding any new package/library, the Agent MUST:
  1. Check if a similar dependency already exists in the project.
  2. Verify the package has no known critical CVEs.
  3. Pin the version in the requirements/dependency file.
  4. Report findings to the user before installing.
- **HARD STOP** if a critical CVE is found — require explicit user approval to proceed.

### 7.6 Breaking-Change Guard
- Any change that modifies a **public API signature**, **database schema**, or **shared data contract** MUST:
  1. Be flagged as a BREAKING CHANGE before implementation.
  2. Require explicit user approval (HARD STOP).
  3. Include a migration plan or backward-compatibility strategy in the implementation plan.

---

## 8. Definition of Done

A task is ONLY complete when ALL of the following are true:

- [ ] Meets defined requirements
- [ ] Code runs without errors
- [ ] All tests pass (including new tests written for this change)
- [ ] No secrets or credentials in code or git history
- [ ] Code has been self-reviewed (Section 7.3 checklist)
- [ ] Developer can explain every part of the generated code
- [ ] Documentation is updated (README.md and/or AI_Usage.md)
- [ ] Changes are committed on the correct feature branch (not main/master)
- [ ] Observability hooks are included for new modules
- [ ] AI contribution is disclosed in AI_Usage.md

---

*Rules file generated from: `Rule chung de su dung AI trong phan.md`*
*Last updated: 2026-09-15*
