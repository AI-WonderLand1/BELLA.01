# Bella Hydra — Multi-Brain AI Architecture

**Status:** Architecture proposal  
**Project:** BELLA.01  
**Primary design decision:** Custom Node.js + TypeScript orchestration, with Python services where the AI ecosystem requires them.

## 1. Product goal

Bella should feel like one AI to the user:

- One chat interface and conversation identity.
- One consistent assistant voice and response style.
- One memory and permission model.
- Multiple specialist models and tools working behind the scenes.
- One coherent answer or completed task returned to the user.

The internal system is called **Hydra**. Hydra is Bella's orchestration layer, not a collection of separate user-facing bots.

## 2. Architecture at a glance

```text
                    BELLA UI / CHAT / VOICE
                              |
                              v
                 BELLA API — Node.js / TypeScript
                              |
                              v
                    HYDRA ORCHESTRATOR
              planning / routing / state / permissions
                              |
               +--------------+--------------+
               |              |              |
               v              v              v
          CODING BRAIN   REASONING BRAIN   VISION BRAIN
          code / tests   planning / logic  image / video
               |              |              |
               +--------------+--------------+
                              |
                   AUDIO / SPEECH SERVICES
                   transcription / synthesis
                              |
                              v
                  CONFESSIONS VERIFICATION
              tests / evidence / policy / quality
                              |
                              v
                   BELLA RESPONSE ASSEMBLER
                              |
                              v
                       ONE BELLA OUTPUT
```

The diagram represents logical services. They do not all need to run as separate servers on day one.

## 3. Technology decisions

| Area | Initial choice | Responsibility |
|---|---|---|
| Main backend and orchestrator | Node.js + TypeScript | Owns task execution and the user-facing API |
| Orchestration logic | Custom TypeScript state machine / workflow engine | Plans, routes, schedules, retries, and combines work |
| HTTP API | Fastify or the existing Node framework | Requests, authentication, streaming |
| Model serving | Compatible inference servers and provider APIs | Runs or exposes model inference |
| Python services | Python, FastAPI when useful | Training, evaluation, and model-specific pipelines |
| Persistent memory | MongoDB; Mem0 integration optional | Conversation and approved long-term memory |
| Queue / job coordination | Redis or an existing queue | Background and long-running work |
| Streaming | Server-Sent Events initially; WebSocket if needed | Incremental progress and responses |
| Verification | TypeScript checks plus dedicated test/security tools | Validates outputs independently |

**Important boundary:** TypeScript owns orchestration. Python is an execution and ML tooling layer, not a second competing conductor. Do not duplicate task-routing logic in both languages.

## 4. Specialist brains

Brains are capability interfaces. A brain can be backed by a model, a tool, or a combination of services.

### Conductor brain

- Interprets the user's request.
- Produces a structured task plan.
- Selects specialist capabilities.
- Establishes dependencies and parallelizable work.
- Tracks execution state and deadlines.
- Requests clarification when essential information is missing.
- Assembles the final user-facing response.

The conductor should not be trusted to declare its own work correct.

### Coding brain

- Generates and edits code.
- Explains repository structure and changes.
- Runs tests, linters, and type checks through controlled tools.
- Returns changed files, test evidence, and known limitations.

### Reasoning brain

- Handles complex analysis, planning, multi-step reasoning, and code review.
- Can independently critique plans and proposed changes.
- Returns conclusions with supporting evidence where available.

### Vision brain

- Image understanding and analysis.
- Image generation through a dedicated image model/service.
- Video generation through a separate video model/service.

Image and video generation are distinct workloads; do not assume one model handles both.

### Audio brain

- Speech-to-text transcription.
- Text-to-speech synthesis.
- Voice activity and audio processing where required.

Whisper-style transcription and Kokoro-style speech synthesis are examples of separate components, not a fixed dependency.

### Security brain

- Reviews proposed actions and generated code.
- Applies authorization and policy checks.
- Flags secrets, unsafe dependencies, and risky operations.
- Cannot replace deterministic security tooling or human approval for high-impact actions.

## 5. Model strategy

The previously discussed Qwen and GPT-OSS models are **candidate backends**, not hard dependencies of the architecture.

- Use a general-purpose model for conversation, multimodal understanding where supported, and routine coding.
- Use a reasoning/tool-use specialist for harder planning, analysis, and verification assistance.
- Route only the relevant task to specialist services; do not send every request to every model.
- Keep model names, endpoints, quantization, context limits, and capabilities in configuration.
- Benchmark model quality, latency, memory use, and license obligations on the actual deployment hardware before selecting production defaults.

Do not assume two GPUs automatically behave as one larger GPU. Model placement, quantization, tensor parallelism, and memory requirements must be tested against the specific inference server and hardware.

## 6. Task execution lifecycle

1. **Receive:** API authenticates the user and accepts text, files, images, or audio.
2. **Normalize:** Convert input into a common request format. Audio transcription is performed when needed.
3. **Plan:** Conductor creates a typed execution plan with task IDs, dependencies, required capabilities, and expected outputs.
4. **Authorize:** Validate user permissions, tool scopes, resource limits, and any required confirmation.
5. **Execute:** Run independent tasks in parallel and dependent tasks in order.
6. **Collect:** Store specialist results, artifacts, logs, and provenance in task state.
7. **Verify:** Run deterministic checks and independent review appropriate to the task.
8. **Repair:** Retry only the failed component with bounded attempts and a clear failure reason.
9. **Assemble:** Conductor combines verified outputs into one response or artifact package.
10. **Deliver:** Stream progress and the final response through the single Bella interface.

A task must be allowed to fail safely. If a specialist is unavailable or verification fails, Bella should disclose the limitation instead of claiming completion.

## 7. Structured task contract

Use validated schemas (for example, Zod) rather than trusting free-form model-generated JSON.

Example plan:

```json
{
  "taskId": "task_001",
  "goal": "Create and preview a French language-learning landing page",
  "steps": [
    {
      "stepId": "build",
      "capability": "coding",
      "dependsOn": [],
      "input": {
        "deliverable": "responsive landing page",
        "securityRequirements": ["validate inputs", "no embedded secrets"]
      }
    },
    {
      "stepId": "translate",
      "capability": "translation",
      "dependsOn": [],
      "input": {
        "language": "fr"
      }
    },
    {
      "stepId": "media",
      "capability": "video",
      "dependsOn": ["build", "translate"],
      "input": {
        "deliverable": "short UI preview"
      }
    },
    {
      "stepId": "voiceover",
      "capability": "speech",
      "dependsOn": ["translate"],
      "input": {
        "language": "fr-FR"
      }
    },
    {
      "stepId": "verify",
      "capability": "verification",
      "dependsOn": ["build", "translate", "media", "voiceover"],
      "input": {
        "checks": ["build", "security", "translation-consistency", "asset-integrity"]
      }
    }
  ]
}
```

This is an illustrative contract, not a claim that these services or actions have already been implemented. Validate every capability, field, and dependency against a server-side allowlist before execution.

## 8. Confessions — verification and quality system

Confessions is Bella's verification layer. It should produce inspectable evidence, not an ungrounded confidence score.

Checks should include:

- **Code:** build, unit tests, integration tests, lint, type checks.
- **Security:** dependency scanning, secret scanning, input validation, authorization checks, and sandboxed execution.
- **Content:** factual claims checked against available evidence; uncertainty reported when evidence is missing.
- **Translation:** compare source and translated strings; use human review for high-stakes or nuanced content.
- **Media:** verify that requested files exist, decode correctly, and meet expected formats and dimensions.
- **Task completion:** confirm each required step completed and each artifact is accessible.

A language model may assist with review, but it is not proof. Use independent validators and bounded retries. Never silently rewrite a failed result and report it as verified.

## 9. Security and execution boundaries

- Enforce authentication and authorization in the API, not in model prompts.
- Treat model output, uploaded files, retrieved documents, and tool results as untrusted input.
- Give each tool the minimum permissions required.
- Require explicit user approval for external writes, deployments, purchases, destructive actions, or other high-impact operations.
- Run generated code in isolated, resource-limited sandboxes without production credentials.
- Keep secrets in a secret manager or protected environment variables; never put tokens in prompts, logs, or training data.
- Apply request, token, time, concurrency, storage, and GPU budgets.
- Record auditable task events without unnecessarily retaining sensitive user content.
- Separate user memory from training datasets. Do not automatically train on conversations or approved outputs.

## 10. Memory design

Memory is a service used by the orchestrator and permitted specialists, not an independent source of authority.

- Short-term state: current task, step outputs, dependencies, and conversation context.
- Long-term memory: only information the product is permitted to retain.
- Retrieval: fetch only relevant records for the current task.
- User controls: provide retention, correction, and deletion workflows.
- Training: require separate consent, privacy review, licensing review, and dataset validation before using any retained content.

MongoDB and Mem0 are candidate components; confirm their operational and licensing fit before adopting them.

## 11. Suggested repository layout

```text
BELLA.01/
├── apps/
│   └── api/
├── packages/
│   ├── hydra-core/
│   ├── contracts/
│   ├── model-gateway/
│   ├── memory/
│   └── confessions/
├── services/
│   ├── coding-brain/
│   ├── reasoning-brain/
│   ├── vision-brain/
│   ├── audio-brain/
│   └── python-ml/
├── configs/
├── evals/
├── docs/
│   ├── BELLA-HYDRA-ARCHITECTURE.md
│   └── ADR/
├── infra/
├── tests/
├── .env.example
└── README.md
```

Adapt this to the existing repository before creating directories. Avoid duplicate services and frameworks if the repo already has equivalents.

## 12. Implementation phases

### Phase 1 — Working foundation

- Inspect the existing BELLA.01 repository and deployment setup.
- Add typed request, task, step, and result contracts.
- Implement the TypeScript API and custom conductor.
- Connect the two selected text-model endpoints through a model gateway.
- Add one coding workflow and one reasoning/review workflow.
- Add task logs, cancellation, timeouts, and bounded retries.
- Add deterministic verification and tests.

**Exit criteria:** A user can submit one request, Hydra can route it to the appropriate model, and Bella returns one response with traceable task results.

### Phase 2 — Reliable agent execution

- Add dependency-aware parallel execution.
- Add sandboxed coding tools.
- Add memory with explicit retention controls.
- Add structured Confessions reports.
- Add evaluation suites for coding, reasoning, latency, and routing.

**Exit criteria:** Failed steps are isolated, logged, and safely retried or reported.

### Phase 3 — Multimodal services

- Add image understanding and image generation.
- Add audio transcription and speech synthesis.
- Add video generation as a separate asynchronous service.
- Add artifact storage, progress streaming, and cancellation.

**Exit criteria:** One user request can combine text, code, image, audio, and video services into one verified result.

### Phase 4 — Training and adaptation

- Build dataset provenance, license, deduplication, privacy, and quality checks.
- Evaluate base models before fine-tuning.
- Train with Python tooling such as Unsloth only where model architecture and hardware are supported.
- Version datasets, adapters, model configurations, and evaluation results.
- Promote a model only when it improves measured performance without unacceptable regressions.

**Exit criteria:** Every trained model has reproducible inputs, documented permissions, and evaluation evidence.

## 13. Ownership, licensing, and limitations

Bella's original orchestration code, interfaces, schemas, and original documentation can be owned by the project subject to the applicable contributor and repository terms.

Using an open model does **not** transfer ownership of its original weights or erase its license conditions. Review the license and usage terms for every model, dataset, dependency, and generated asset before commercial distribution. Preserve required notices and comply with any use restrictions.

The Hydra architecture is a design proposal. It does not itself guarantee model quality, factual accuracy, security, performance, or commercial rights. Those require implementation, testing, operational controls, and license review.

## 14. Current decision record

- **User experience:** one Bella assistant, one chat and voice identity.
- **Architecture:** hierarchical multi-agent system named Hydra.
- **Main orchestrator:** custom Node.js + TypeScript.
- **Python role:** specialist ML, training, and model-specific services.
- **Coordination:** one TypeScript execution graph; no duplicated Python conductor.
- **Verification:** Confessions, combining deterministic checks and independent review.
- **Model selection:** configurable candidates; benchmark before production commitment.
- **Deployment:** not specified by this document; do not deploy or change production infrastructure as part of the architecture documentation task.

## 15. Immediate next engineering task

Inspect BELLA.01's existing files, scripts, runtime, and deployment configuration. Then prepare a minimal-diff implementation plan for Phase 1. Do not overwrite existing application architecture or deploy changes without explicit approval.
