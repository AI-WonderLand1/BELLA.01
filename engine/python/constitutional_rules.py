"""Constitutional Rules for Wonderland AI Evaluation."""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# CONSTITUTIONAL RULES
# ────────────────────────────────────────────────────────────────────
CONSTITUTIONAL_RULES: List[Dict[str, str]] = [
    {"id": "rule_01", "name": "Privacy Protection", "pattern": r"(?i)(password|secret|key|token)\s*[:=]\s*[\"\']?[a-zA-Z0-9]{8,}[\"\']?", "description": "Do not expose passwords, secrets, keys, or tokens in AI output"},
    {"id": "rule_02", "name": "PII Restriction", "pattern": r"(?i)(ssn|social.security|bank.account|credit.card)\s*[:=]\s*[\"\'][^\s]{6,}[\"\']?", "description": "Do not expose personally identifiable information"},
    {"id": "rule_03", "name": "Harmful Content", "pattern": r"(?i)(how to kill|how to harm|weapon|explosive|drugs)", "description": "Do not generate harmful or dangerous content"},
    {"id": "rule_04", "name": "Copyright", "pattern": r"(?i)(copyrighted|do not redistribute|all rights reserved)", "description": "Do not generate copyrighted content without attribution"},
    {"id": "rule_05", "name": "Security Boundaries", "pattern": r"(?i)(exec\(|eval\(|system\(|os\.system)", "description": "Do not generate code that executes system commands"},
    {"id": "rule_06", "name": "Confidentiality", "pattern": r"(?i)(confidential|non.disclosure|nda)", "description": "Do not disclose confidential or NDA-protected information"},
]

# ────────────────────────────────────────────────────────────────────
# CONSTITUTIONAL EVALUATOR
# ──────────────────────────────────────────────────────────────────


def evaluate_against_constitution(text: str) -> List[Dict[str, str]]:
    """Evaluate AI output against Wonderland Constitutional Rules."""
    violations: List[Dict[str, str]] = []
    for rule in CONSTITUTIONAL_RULES:
        try:
            if re.search(rule["pattern"], text):
                violations.append({"ruleId": rule["id"], "description": rule["description"]})
        except Exception:
            continue
    return violations


# ────────────────────────────────────────────────────────────────────
# CONFESSIONS SYSTEM
# ──────────────────────────────────────────────────────────────────


class AgentName:
    """Supported agent names for type safety."""
    ALICE = "Alice"
    SIMPLE_RICK = "Simple Rick"
    SPIRIT_GUIDE = "Spirit Guide"


class ConfessionEventFactory:
    """Factory for creating confession events."""

    @staticmethod
    def create(operationId: str, eventId: str, sequence: int,
               agentName: AgentName = None, confession: Dict = None,
               language: str = "en", timestamp: str = None) -> Dict:
        """Create a confession event dictionary."""
        import datetime
        result = {
            "operationId": operationId,
            "eventId": eventId,
            "eventType": "CONFESSION",
            "agentName": agentName.value if isinstance(agentName, AgentName) else (agentName or "Simple Rick"),
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "sequence": sequence,
            "language": language,
            "confession": confession or {"text": "", "language": language, "agentName": None},
        }
        return result

    @staticmethod
    def from_user_prompt(user_prompt: str, agentName: AgentName = AgentName.ALICE) -> Dict:
        """Create a confession event from a user prompt."""
        import uuid
        return ConfessionEventFactory.create(
            operationId="gen_" + uuid.uuid4().hex[:8],
            eventId="event_" + uuid.uuid4().hex[:8],
            sequence=1,
            agentName=agentName,
            confession={"text": user_prompt, "language": "en", "agentName": agentName.value},
        )


# ────────────────────────────────────────────────────────────────────
# 3-BRAIN ARCHITECTURE
# ──────────────────────────────────────────────────────────────────


class BrainType:
    """Types of specialist brains in the 3-brain system."""
    CONDUCTOR = "conductor"
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"


class ThreeBrainSystem:
    """Three-brain AI system operating as one unit."""

    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        self.brain1_conductor = ConductorBrain()
        self.brain2_coding = CodingBrain()
        self.brain3_reasoning = ReasoningBrain()
        self.constitutional_evaluator = evaluate_against_constitution

    def process_request(self, user_prompt: str) -> Dict[str, Any]:
        """Process a user request through all three brains as one unit."""
        plan = self.brain1_conductor.plan_request(user_prompt)
        generation_result = self.brain2_coding.generate_fullstack(plan)
        constitutional_violations = self.constitutional_evaluator(
            generation_result.get("generated_code", "")
        )
        reasoning_result = self.brain3_reasoning.validate_generation(
            generation_result, plan
        )
        integrated_result = self._integrate_results(
            plan, generation_result, constitutional_violations, reasoning_result
        )
        return integrated_result

    def _integrate_results(self, plan, generation, violations, reasoning):
        """Integrate outputs from all three brains."""
        return {
            "success": len(violations) == 0 and reasoning.get("valid", False),
            "plan": plan,
            "frontend": generation.get("frontend"),
            "backend": generation.get("backend"),
            "constitutional_violations": violations,
            "reasoning_valid": reasoning.get("valid", False),
            "reasoning_feedback": reasoning.get("feedback", ""),
            "generated_code": generation.get("generated_code", ""),
            "confession_event": self._create_confession_event(plan, generation),
            "ready_for_deployment": len(violations) == 0 and reasoning.get("valid", False),
        }

    def _create_confession_event(self, plan, generation):
        """Create confession event for this generation."""
        return ConfessionEventFactory.create(
            operationId="gen_" + os.urandom(4).hex()[2:],
            eventId="event_" + os.urandom(4).hex()[2:],
            sequence=1,
            agentName=AgentName.ALICE,
            confession={
                "text": "Generated " + plan.get("app_type", "application") + ": " + plan.get("description", "")[:50] + "...",
                "language": "en",
                "agentName": "Alice",
                "constitutional_check": {
                    "passed": len(self.constitutional_evaluator(generation.get("generated_code", ""))) == 0,
                    "violations": self.constitutional_evaluator(generation.get("generated_code", "")),
                },
            },
        )


class ConductorBrain:
    """Brain 1: Conductor - Plans and routes requests."""

    def plan_request(self, user_prompt: str) -> Dict[str, Any]:
        """Analyze user prompt and plan the response."""
        prompt_lower = user_prompt.lower()
        if any(k in prompt_lower for k in ["todo", "task", "management"]):
            app_type = "todo_application"
        elif any(k in prompt_lower for k in ["portfolio", "showcase", "personal"]):
            app_type = "portfolio_website"
        elif any(k in prompt_lower for k in ["blog", "article", "post"]):
            app_type = "blog_platform"
        elif any(k in prompt_lower for k in ["shop", "store", "ecommerce", "sell"]):
            app_type = "ecommerce_store"
        else:
            app_type = "general_web_application"
        features = []
        if any(k in prompt_lower for k in ["database", "sql", "db"]):
            features.append("database")
        if any(k in prompt_lower for k in ["responsive", "mobile", "viewport"]):
            features.append("responsive")
        if any(k in prompt_lower for k in ["auth", "login", "signup"]):
            features.append("authentication")
        if not features:
            features = ["basic"]
        return {
            "app_type": app_type,
            "description": user_prompt,
            "features": features,
            "user_intent": user_prompt,
        }


class CodingBrain:
    """Brain 2: Coding - Generates frontend and backend code."""

    def generate_fullstack(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete frontend + backend from plan."""
        app_type = plan.get("app_type", "general_web_application")
        description = plan.get("description", "")
        features = plan.get("features", ["basic"])
        backend = self._generate_backend(plan)
        frontend = self._generate_frontend(plan)
        generated_code = self._combine_code(frontend, backend)
        return {
            "frontend": frontend,
            "backend": backend,
            "generated_code": generated_code,
            "app_type": app_type,
        }

    def _generate_backend(self, plan):
        """Generate backend code (FastAPI)."""
        app_name = "GeneratedApp"
        db_table = "users"
        code = (
            '"""Generated backend by BELLA.01 3-Brain AI."""\n'
            'from fastapi import FastAPI, HTTPException, Depends\n'
            'from fastapi.middleware.cors import CORSMiddleware\n'
            'from pydantic import BaseModel, Field\n'
            'from typing import Optional, List\n'
            'import sqlite3\n'
            'import uuid\n'
            'from datetime import datetime, timezone\n'
            ''
            'app = FastAPI('
            '    title="' + app_name + '",'
            '    description="Generated by BELLA.01 3-Brain AI System",'
            '    version="0.1.0"'
            ")"
            ""
            'app.add_middleware('
            '    CORSMiddleware,'
            '    allow_origins=["*"]'
            '    allow_credentials=True,'
            '    allow_methods=["*"]'
            '    allow_headers=["*"]'
            ")"
            ""
            'DATABASE = "database.db"'
            ""
            'def get_db():'
            '    conn = sqlite3.connect(DATABASE)'
            '    conn.row_factory = sqlite3.Row'
            '    try:'
            '        yield conn'
            '        conn.commit()'
            '    except Exception:'
            '        conn.rollback()'
            '        raise'
            '    finally:'
            '        conn.close()'
            ''
            'def init_db():'
            '    with get_db() as conn:'
            '        c = db_table'
            '        cursor = conn.cursor()'
            '        cursor.execute('
            '            "CREATE TABLE IF NOT EXISTS " + c + " (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"'
            ")"
            ""
            "init_db()"
            ""
            'class ' + app_name.title().replace(" ", "") + 'Base(BaseModel):'
            '    id: str = Field(default_factory=lambda: str(uuid.uuid4()))'
            '    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())'
            '    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())'
            '    model_config = {"from_attributes": True}'
        )
        return {"code": code, "type": "FastAPI"}

    def _generate_frontend(self, plan):
        """Generate frontend code (Vanilla HTML/CSS/JS)."""
        app_name = "GeneratedApp"
        html = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            f'<title>' + app_name + '</title><link rel="stylesheet" href="style.css">'
            '<style>:root{--primary:#2d3748;--secondary:#4a5568;--bg:#f7fafc;--text:#1a202c;}'
            '*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'
            'background:var(--bg);color:var(--text);min-height:100vh}'
            'header{background:var(--primary);color:white;padding:2rem;text-align:center}'
            'main{max-width:800px;margin:0auto;padding:2rem}h1{font-size:2rem;margin-bottom:1rem;text-align:center}'
            'p{line-height:1.6;margin-bottom:1.5rem}@media(max-width:600px){main{padding:1rem}}</style>'
            '</head><body><header><h1>Welcome to ' + app_name + '</h1><p>Generated by BELLA.01 3-Brain AI System</p></header>'
            '<main><section class="intro"><h2>' + plan.get("description", "Application") + '</h2>'
            '<p>This application was automatically generated by AI.</p></section></main></body></html>'
        )
        css = (
            '/* Generated by BELLA.01 */:root{--primary:#2d3748;--secondary:#4a5568;--accent-color:#48bb78;--bg-color:#f7fafc;--text-color:#1a202c;}*'
            '{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'
            'background-color:var(--bg-color);color:var(--text-color);min-height:100vh;line-height:1.6}'
            'header{background-color:var(--primary);color:white;padding:2rem;text-align:center;margin-bottom:2rem}'
            'main{max-width:800px;margin:0auto;padding:0 1rem}h1{font-size:2rem;margin-bottom:1rem;text-align:center}'
            'p{line-height:1.6;margin-bottom:1.5rem}@media(max-width:600px){header{padding:1rem}h1{font-size:1.5rem}}'
        )
        return {"code": html, "css": css, "type": "Vanilla HTML/CSS/JS"}

    @staticmethod
    def _rand_color():
        import random
        return "#" + format(random.randint(0, 0xFFFFFF), "06x")

    @staticmethod
    def _rand_color_light():
        import random
        shades = ["f7fafc", "edf2f7", "e2e8f0", "cbd5e0", "a0aec0"]
        return random.choice(shades)

    def _combine_code(self, frontend, backend):
        """Combine frontend and backend into comprehensive code output."""
        fc = frontend.get("code", "")
        bc = backend.get("code", "")
        return (
            "// ==========================================\n"
            "// BELLA.01 3-BRAIN AI GENERATED FULLSTACK\n"
            "// Frontend + Backend Integration\n"
            "// ==========================================\n"
            "// BACKEND (FastAPI)\n"
            + bc
            + "\n// FRONTEND\n"
            + fc
            + "\n// ==========================================\n"
            "// INSTRUCTIONS:\n"
            "// 1. Place backend code in /backend/app.py\n"
            "// 2. Place frontend code in /frontend/index.html\n"
            "// 3. Run: cd backend && uvicorn app:app --reload\n"
            "// 4. Open frontend in browser\n"
            "// ==========================================\n"
        )


class ReasoningBrain:
    """Brain 3: Reasoning - Evaluates, validates, checks constitutionality."""

    def validate_generation(self, generation: Dict, plan: Dict) -> Dict[str, Any]:
        """Validate the generated code and content."""
        feedback: List[str] = []
        valid = True
        code = generation.get("generated_code", "")
        violations = evaluate_against_constitution(code)
        if violations:
            valid = False
            feedback.append("Constitutional violations found: " + str(len(violations)) + " rules broken")
        if not generation.get("frontend"):
            valid = False
            feedback.append("Missing frontend generation")
        if not generation.get("backend"):
            valid = False
            feedback.append("Missing backend generation")
        if not plan.get("description"):
            valid = False
            feedback.append("No clear plan description")
        return {
            "valid": valid,
            "feedback": " | ".join(feedback) if feedback else "Generation validated successfully",
            "constitutional_violations": violations,
        }


# ────────────────────────────────────────────────────────────────────
# PROMPT INTERFACE
# ────────────────────────────────────────────────────────────────────


def chat_with_3_brains(user_input: str, hf_token: str = None) -> Dict[str, Any]:
    """Entry point for chatting with the 3-brain system."""
    system = ThreeBrainSystem(hf_token=hf_token)
    return system.process_request(user_input)


# ────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("🧠 BELLA.01 3-BRAIN AI SYSTEM")
    print("=" * 70)
    print("\nDescribe the application you want to build:")
    print("> ", end="")

    test_prompts = [
        "A todo application with task management, SQLite backend, and React frontend",
        "A personal portfolio website with about section, project gallery, and contact form",
        "A blog platform with user authentication, post editing, and comment system",
    ]

    for prompt in test_prompts:
        print(f"\n🔍 Processing: {prompt}")
        result = chat_with_3_brains(prompt)

        print(f"\n✅ Success: {result['success']}")
        print("   App Type: " + result['plan']['app_type'])
        print("   Violations: " + str(len(result['constitutional_violations'])))
        print("   Valid: " + str(result['reasoning_valid']))
        print("   Frontend: " + ("Generated" if result['frontend'] else "None"))
        print("   Backend: " + ("Generated" if result['backend'] else "None"))
        print("   Ready: " + str(result['ready_for_deployment']))

        if result['constitutional_violations']:
            print("   🚫 Violations:")
            for v in result['constitutional_violations']:
                print("     - " + v['ruleId'] + ": " + v['description'])

    print("\n" + "=" * 70)
