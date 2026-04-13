# TriGate — Autonomous Implementation Instructions

## Behaviour rules
- Work through milestones M1→M5 in order, exactly as defined in trigate_impl_spec.docx
- Complete ALL tasks in a milestone before moving to the next
- After every milestone: run the acceptance tests, git commit, git push
- Do NOT ask for permission at any step — proceed autonomously
- If a task is blocked, write BLOCKED: [task id] [reason] to blocked.md, skip it, continue
- Never wait for user input

## After each milestone
```bash
git add -A
git commit -m "feat: complete M[N] — [milestone name]"
git push origin main
```

## Git config
- Branch: main
- Commit style: "feat: complete M1 — gateway skeleton"
- Tag each milestone: git tag v0.1.0 after M1, v0.2.0 after M2, etc.

## Start here
Read trigate_impl_spec.docx. Begin M1.1. Do not stop until M5.12 is complete and pushed.
