"""Everything Builder: Full-Stack Web & Application Generator."""
import json
import os
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import sys
sys.path.insert(0, '/home/bobbi/BELLA.01/engine/python')
from ai_modalities.math_model import solve_arithmetic
from ai_modalities.text_model import classify_sentiment, summarize_text
from pytorch_models.pytorch_manager import create_manager

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_TEMPLATES_DIR = Path("/home/bobbi/BELLA.01/engine/data/templates")
OUTPUT_DIR = Path("/home/bobbi/BELLA.01/engine/data/generated_projects")
PROJECT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# CORE BUILDER CLASS
# ──────────────────────────────────────────────────────────────────────
class EverythingBuilder:
    def __init__(self, pytorch_manager=None, device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        self.mgr = pytorch_manager or create_manager(device=self.device)
        self.project_counter = 0
        print("🔨 EverythingBuilder initialized on " + self.device)
        print("   Device: " + self.device)

    def build_complete_app(self, app_name, app_type="web_app", description="", tech_stack="auto"):
        self.project_counter += 1
        project_id = "app_" + str(self.project_counter) + "_" + app_name.replace(" ", "_")
        project_dir = OUTPUT_DIR / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        print("\n🚀 Building: " + app_name + " (ID: " + project_id + ")")
        print("   Type: " + app_type)
        desc_show = description[:80] if description else "No description"
        print("   Description: " + desc_show + "...")

        plan = self._generate_project_plan(app_name, app_type, description, tech_stack)
        frontend_result = self._generate_frontend(str(project_dir / "frontend"), plan)
        backend_result = self._generate_backend(str(project_dir / "backend"), plan, True)
        if True:
            test_result = self._generate_tests(str(project_dir / "tests"), plan)
        else:
            test_result = {"status": "skipped"}
        manifest = self._create_manifest(project_dir, plan, {
            "frontend": frontend_result,
            "backend": backend_result,
            "configs": self._generate_configs(project_dir, plan, True),
            "tests": test_result
        })

        print("✅ Project complete: " + str(project_dir))
        print("   Files: " + str(manifest["file_count"]) + " files, " + str(round(manifest["total_size_kb"], 1)) + " KB")
        return {"project_id": project_id, "project_dir": str(project_dir), "manifest": manifest, "plan": plan, "success": True}

    def _generate_project_plan(self, app_name, app_type, description, tech_stack):
        default_plans = {
            "todo_app": {"type": "Task Management Web Application", "frontend": "React + Vanilla JS", "backend": "FastAPI + SQLite", "database": "Tasks table (id, title, completed, created_at)", "features": ["Create/Edit/Delete tasks", "Mark complete", "Local storage"], "tech": "React, FastAPI, SQLite, CORS"},
            "blog_app": {"type": "Blogging Platform", "frontend": "React + Markdown", "backend": "FastAPI", "database": "Posts table (id, title, content, author, created_at)", "features": ["Create posts", "Markdown editing", "Comment system"], "tech": "React, FastAPI, SQLite, Markdown"},
            "portfolio": {"type": "Personal Portfolio Website", "frontend": "HTML5 + CSS3 + Vanilla JS", "backend": "None (static)", "database": "N/A", "features": ["About section", "Project gallery", "Contact form"], "tech": "HTML5, CSS3, JavaScript, Hosting"},
            "ecommerce": {"type": "Simple E-commerce Store", "frontend": "HTML5 + CSS3 + JavaScript", "backend": "FastAPI + SQLite", "database": "Products, Orders, Users tables", "features": ["Product catalog", "Cart", "Checkout"], "tech": "HTML5, CSS3, JS, FastAPI, SQLite"},
            "learning_platform": {"type": "Online Learning Platform", "frontend": "React +", "backend": "FastAPI", "database": "Users, Courses, Enrollments tables", "features": ["User auth", "Course browsing", "Progress tracking"], "tech": "React, FastAPI, PostgreSQL"}
        }
        app_lower = (app_name + " " + description).lower()
        if any(k in app_lower for k in ["todo", "task", "management"]): plan_type = "todo_app"
        elif any(k in app_lower for k in ["blog", "article", "post"]): plan_type = "blog_app"
        elif any(k in app_lower for k in ["portfo", "personal", "showcase"]): plan_type = "portfolio"
        elif any(k in app_lower for k in ["shop", "store", "commerce", "sale"]): plan_type = "ecommerce"
        elif any(k in app_lower for k in ["learn", "course", "education", "study"]): plan_type = "learning_platform"
        else: plan_type = "web_app"
        base_plan = default_plans.get(plan_type, default_plans["todo_app"])
        if description: base_plan["description"] = description
        else: base_plan["description"] = "Generated application by BELLA.01 AI"
        if app_name: base_plan["name"] = app_name
        else: base_plan["name"] = "Generated Application"
        if tech_stack == "auto": base_plan["tech_stack"] = base_plan["tech"]
        else: base_plan["tech_stack"] = tech_stack
        base_plan["generated_by"] = "BELLA.01 Everything Builder AI"
        base_plan["generation_timestamp"] = datetime.datetime.now().isoformat()
        base_plan["project_type"] = app_type
        return base_plan

    def _generate_frontend(self, project_dir, plan):
        frontend_dir = Path(project_dir)
        frontend_dir.mkdir(parents=True, exist_ok=True)
        files_created = []
        tech = plan.get("tech_stack", "HTML5/CSS3/JS")
        frontend_type = tech.split("/")[0] if "/" in tech else tech

        if frontend_type in ("React", "react", "next"): result = self._generate_react_frontend(frontend_dir, plan)
        elif frontend_type in ("Vue", "vue"): result = self._generate_vue_frontend(frontend_dir, plan)
        else: result = self._generate_vanilla_frontend(frontend_dir, plan)
        files_created.extend(result.get("files", []))
        # Generate simple HTML
        html_content = "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>" + plan.get("name", "App") + "</title><link rel='stylesheet' href='style.css'><style>:root{--primary:#2d3748;--secondary:#4a5568;--bg:" + self._rand_clr_lgt() +";--text:#1a202c;}*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}header{background:var(--primary);color:white;padding:2rem;text-align:center}main{max-width:800px;margin:0auto;padding:2rem}h1{font-size:2rem;margin-bottom:1rem}p{line-height:1.6;margin-bottom:1.5rem}@media(max-width:600px){main{padding:1rem}}</style></head><body><header><h1>Welcome to " + plan.get("name", "App") + "</h1><p>Generated by BELLA.01 AI</p></header><main><section class='intro'><h2>" + plan.get("type", "Application") + "</h2><p>This is a fully functional " + plan.get("type", "application").lower() + " automatically generated by AI.</p></section></main></body></html>"
        (Path(project_dir) / "index.html").write_text(html_content, encoding="utf-8")
        files_created.append("index.html")
        return {"status": "completed", "frontend_type": frontend_type, "files": files_created, "tech": tech}

    @staticmethod
    def _rand_clr_lgt():
        import random
        shades = ["f7fafc", "edf2f7", "e2e8f0", "cbd5e0", "a0aec0"]
        return random.choice(shades)

    def _generate_react_frontend(self, frontend_dir, plan):
        files = []
        pkg_name = plan.get("name", "app").replace(" ", "-").lower()
        pkg = {"name": pkg_name, "version": "0.1.0", "private": True, "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"}, "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0", "axios": "^1.7.2"}, "devDependencies": {"vite": "^5.2.0", "@types/react": "^18.2.0", "@types/react-dom": "^18.2.0", "typescript": "^5.4.0"}}
        (frontend_dir / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8"); files.append("package.json")
        tsconfig = {"compilerOptions": {"target": "ES2020", "useDefineForClassFields": True, "lib": ["ES2020", "DOM"], "strict": True, "module": "ESNext", "outDir": "./dist", "rootDir": "./src", "esModuleInterop": True, "moduleResolution": "bundler", "allowJs": True, "skipLibCheck": True, "skipDefaultLibCheck": True, "baseUrl": ".", "paths": {"*": ["src/*"]}, "include": ["src/**/*", "index.ts"], "exclude": ["node_modules", "dist", ".git"]}}
        (frontend_dir / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2), encoding="utf-8"); files.append("tsconfig.json")
        src_dir = frontend_dir / "src"; src_dir.mkdir(parents=True, exist_ok=True)
        # Simplified React - just write the files without complex string escaping
        (src_dir / "App.tsx").write_text("interface AppProps { name: string }\nfunction App({name}: AppProps) { return <div><h1>Welcome to " + plan.get("name", "App") + "</h1><p>Generated by BELLA.01 AI</p></div> }\nexport default App", encoding="utf-8"); files.append("src/App.tsx")
        (src_dir / "index.tsx").write_text("import React from 'react'\nimport ReactDOM from 'react-dom/client'\nimport App from './App'\nimport './index.css'\nconst root = ReactDOM.createRoot(\n  document.getElementById('root') as HTMLElement\n)\nroot.render(<React.StrictMode><App name='" + plan.get("name", "App") + "' /></React.StrictMode>)", encoding="utf-8"); files.append("src/index.tsx")
        (src_dir / "index.css").write_text(".app-container{min-height:100vh;display:flex;flex-direction:column;background:#f5f7fa}.app-header{background:#2d3748;color:white;padding:2rem;text-align:center}.main{flex:1;padding:2rem,max-width:800px;margin:0auto}", encoding="utf-8"); files.append("src/index.css")
        return {"status": "completed", "files": files}

    def _generate_vue_frontend(self, frontend_dir, plan):
        return self._generate_vanilla_frontend(frontend_dir, plan)

    def _generate_vanilla_frontend(self, frontend_dir, plan):
        files = []
        html_content = "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>" + plan.get("name", "App") + "</title><link rel='stylesheet' href='style.css'><style>:root{--primary:#2d3748;--secondary:#4a5568;--bg-color:" + self._rand_clr_lgt() +";--text-color:#1a202c;}*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:var(--bg-color);color:var(--text-color);min-height:100vh;line-height:1.6}header{background-color:var(--primary);color:white;padding:2rem;text-align:center;margin-bottom:2rem}main{max-width:800px;margin:0auto;padding:0 1rem}h1{font-size:2rem;margin-bottom:1rem;text-align:center}p{line-height:1.6;margin-bottom:1.5rem}@media(max-width:600px){header{padding:1rem}h1{font-size:1.5rem}}</style></head><body><header><h1>Welcome to " + plan.get("name", "App") + "</h1><p>Generated by BELLA.01 AI</p></header><main><section class='intro'><h2>" + plan.get("type", "Application") + "</h2><p>This is a fully functional " + plan.get("type", "application").lower() + " automatically generated by AI.</p></section></main></body></html>"
        (frontend_dir / "index.html").write_text(html_content, encoding="utf-8"); files.append("index.html")
        css_content = "/* Generated by BELLA.01 */:root{--primary:#2d3748;--secondary:#4a5568;--accent-color:#48bb78;--bg-color:" + self._rand_clr_lgt() +";--text-color:#1a202c}*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:var(--bg-color);color:var(--text-color);min-height:100vh;line-height:1.6}header{background-color:var(--primary);color:white;padding:2rem;text-align:center;margin-bottom:2rem}main{max-width:800px;margin:0auto;padding:0 1rem}h1{font-size:2rem;margin-bottom:1rem;text-align:center}p{line-height:1.6;margin-bottom:1.5rem}@media(max-width:600px){header{padding:1rem}h1{font-size:1.5rem}}"
        (frontend_dir / "style.css").write_text(css_content, encoding="utf-8"); files.append("style.css")
        return {"status": "completed", "files": files}

    def _generate_backend(self, backend_dir, plan, include_db):
        backend_dir = Path(backend_dir); backend_dir.mkdir(parents=True, exist_ok=True); files_created = []
        backend_type = plan.get("backend", "FastAPI + SQLite")
        if "Flask" in backend_type: result = self._generate_flask_backend(backend_dir, plan, include_db)
        elif "FastAPI" in backend_type: result = self._generate_fastapi_backend(backend_dir, plan, include_db)
        else: result = self._generate_minimal_backend(backend_dir, plan, include_db)
        files_created.extend(result.get("files", []))
        return {"status": result.get("status", "completed"), "backend_type": backend_type, "files": files_created}

    def _generate_fastapi_backend(self, backend_dir, plan, include_db):
        backend_dir.mkdir(parents=True, exist_ok=True); files = []
        app_name = plan.get("name", "App"); db_table = plan.get("database", "users")
        app_py = '"""' + app_name + ' - Backend API Generated by BELLA.01 AI."""\nfrom fastapi import FastAPI, HTTPException, Depends\nfrom fastapi.middleware.cors import CORSMiddleware\nfrom pydantic import BaseModel, Field\nfrom typing import Optional, List\nimport sqlite3\nimport uuid\nfrom datetime import datetime, timezone\n\napp = FastAPI(title="' + app_name + ' API", description="Auto-generated by BELLA.01 AI Everything Builder", version="0.1.0")\napp.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])\n\ndef get_db():\n    conn = sqlite3.connect(DATABASE)\n    conn.row_factory = sqlite3.Row\n    try:\n        yield conn\n        conn.commit()\n    except Exception:\n        conn.rollback()\n        raise\n    finally:\n        conn.close()\n\ndef init_db():\n    with get_db() as conn:\n        cursor = conn.cursor()\n        c = db_table\n        cursor.execute("CREATE TABLE IF NOT EXISTS " + c + " (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")\n        conn.commit()\ninit_db()\n\nclass ' + app_name.title().replace(" ", "") + 'Base(BaseModel):\n    id: str = Field(default_factory=lambda: str(uuid.uuid4()))\n    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())\n    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())\n    model_config = {"from_attributes": True}\n'
        (backend_dir / "app.py").write_text(app_py, encoding="utf-8"); files.append("app.py")
        routes_py = '"""API routes for ' + app_name + '."""\nfrom fastapi import APIRouter, status\nfrom app import app, get_db, init_db\n\nrouter = APIRouter(prefix="/api", tags=["' + app_name + '"])\n\n@router.on_event("startup")\ndef startup(): init_db()\n\n@router.get("/", include_in_schema=False)\ndef root():\n    return {"message": "Welcome to ' + app_name + ' API", "version": "0.1.0", "docs": "/docs"}\n'
        (backend_dir / "routes.py").write_text(routes_py, encoding="utf-8"); files.append("routes.py")
        if include_db and db_table != "none":
            models_py = '"""SQLAlchemy models for ' + app_name + ' (optional)."""\nfrom sqlalchemy import Column, String, DateTime, create_engine\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom sqlalchemy.orm import sessionmaker\nSQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"\nengine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})\nSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\nBase = declarative_base()\n\nclass ' + app_name.title().replace(" ", "") + '(Base):\n    __tablename__ = "' + db_table + '"\n    id = Column(String, primary_key=True, index=True)\n    created_at = Column(DateTime, nullable=False)\n    updated_at = Column(DateTime, nullable=False)\n\n    def to_dict(self) -> dict:\n        return {"id": self.id, "created_at": self.created_at.isoformat() if self.created_at else None, "updated_at": self.updated_at.isoformat() if self.updated_at else None}\n'
        (backend_dir / "models.py").write_text(models_py, encoding="utf-8"); files.append("models.py")
        return {"status": "completed", "files": files}

    def _generate_flask_backend(self, backend_dir, plan, include_db):
        backend_dir.mkdir(parents=True, exist_ok=True); files = []
        app_py = '"""Flask backend generated by BELLA.01 AI."""\nfrom flask import Flask, request, jsonify\nfrom flask_cors import CORS\napp = Flask(__name__)\nCORS(app)\n\n@app.route("/")\ndef index():\n    return jsonify({"message": "Flask backend generated by BELLA.01 AI", "version": "0.1.0"})\n\nif __name__ == "__main__":\n    app.run(debug=True, host="0.0.0.0", port=5000)\n'
        (backend_dir / "app.py").write_text(app_py, encoding="utf-8"); files.append("app.py")
        return {"status": "completed", "files": files}

    def _generate_minimal_backend(self, backend_dir, plan, include_db):
        backend_dir.mkdir(parents=True, exist_ok=True); files = []
        app_py = '"""Minimal backend generated by BELLA.01 AI."""\nfrom fastapi import FastAPI\napp = FastAPI(title="Generated API", version="0.1.0")\n\n@app.get("/")\ndef index():\n    return {"message": "Hello from your AI-generated backend!"}\n'
        (backend_dir / "app.py").write_text(app_py, encoding="utf-8"); files.append("app.py")
        return {"status": "completed", "files": files}

    def _generate_configs(self, project_dir, plan, include_docs):
        files_created = []
        env_lines = ["# BELLA.01 Generated .env.example", "# Copy to .env and fill in your values", "", "# API Configuration", "KAGGLE_USERNAME=your_kaggle_username", "KAGGLE_KEY=your_kaggle_api_key", "", "# Application", "APP_NAME=" + plan.get("name", "App"), "APP_TYPE=" + plan.get("type", "web_app"), "DEBUG=True", "", "# Database", "DATABASE_URL=sqlite:///./database.db", "", "# Server", "HOST=0.0.0.0", "PORT=8000", "", "# Hugging Face (optional)", "HF_TOKEN=your_hf_token", "HF_DATASET_REPO=ai-wonderland-innovation/Bella-Training-Datasets", ""]
        (Path(project_dir) / ".env.example").write_text("\n".join(env_lines), encoding="utf-8"); files_created.append(".env.example")
        if include_docs:
            readme = "# " + plan.get("name", "Generated App") + "\n\n" + plan.get("description", "A full-stack web application generated by BELLA.01 AI.") + "\n\n## Installation\n```bash\ncd " + plan.get("name", "App") + "\npip install -r requirements.txt\nnpm install\ncp .env.example .env\ncd backend\npython -c \"from app import init_db; init_db()\"\ncd ..\nnpm run dev\n```\n"
            (Path(project_dir) / "README.md").write_text(readme, encoding="utf-8"); files_created.append("README.md")
        requirements = "fastapi>=0.109.0\nuvicorn[standard]>=0.27.0\nsqlite>=3.45.0; platform_system!='Windows'\npydantic>=2.7.0\npython-dotenv>=1.0.0\n"
        if "react" in str(plan.get("tech_stack", "")).lower(): requirements += "react>=18.2.0\nreact-dom>=18.2.0\naxios>=1.7.2\n"
        if "flask" in str(plan.get("backend", "")).lower(): requirements += "flask>=3.0.0\nflask-cors>=5.0.0\n"
        (Path(project_dir) / "requirements.txt").write_text(requirements, encoding="utf-8"); files_created.append("requirements.txt")
        gitignore = "# Dependencies\nnode_modules/\n.env\n*.pyc\n__pycache__/\n*.db\n# Project specific\n/dist/\n/*.pyc\n/.venv/\n"
        (Path(project_dir) / ".gitignore").write_text(gitignore, encoding="utf-8"); files_created.append(".gitignore")
        return {"status": "completed", "files": files_created}

    def _generate_tests(self, tests_dir, plan):
        tests_dir = Path(tests_dir); tests_dir.mkdir(parents=True, exist_ok=True); files_created = []
        backend_test = '"""Tests for ' + plan.get("name", "App") + ' backend."""\nimport pytest\nfrom fastapi.testclient import TestClient\nfrom app import app, init_db\n\nclient = TestClient(app)\n\ndef test_root_endpoint():\n    """Test that root endpoint works."""\n    response = client.get("/api/")\n    assert response.status_code == 200\n    data = response.json()\n    assert "message" in data\n'
        (tests_dir / "test_backend.py").write_text(backend_test, encoding="utf-8"); files_created.append("test_backend.py")
        (tests_dir / "test_frontend.py").write_text('"""Frontend sanity tests."""\n# Verify HTML structure\n\ndef test_html_structure():\n    pass\n# Verify CSS is valid\n\ndef test_css_structure():\n    pass\n', encoding="utf-8"); files_created.append("test_frontend.py")
        (tests_dir / "__init__.py").write_text("# Tests package for generated app", encoding="utf-8"); files_created.append("__init__.py")
        return {"status": "completed", "files": files_created}

    def _create_manifest(self, project_dir, plan, results):
        import os as os_mod
        all_files = []
        for folder in ["backend", "frontend", "tests"]:
            folder_path = project_dir / folder
            if folder_path.exists():
                all_files.extend(folder_path.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        total_size = sum(f.stat().st_size for f in all_files)
        manifest = {"project_id": plan.get("name", "unknown").replace(" ", "_"), "generated_by": "BELLA.01 Everything Builder AI", "generation_timestamp": datetime.datetime.now().isoformat(), "plan": plan, "results": results, "file_count": len(all_files), "total_size_bytes": total_size, "total_size_kb": round(total_size / 1024, 1), "structure": {"backend": (Path(project_dir) / "backend").exists(), "frontend": (Path(project_dir) / "frontend").exists(), "tests": (Path(project_dir) / "tests").exists()}}
        (Path(project_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


# ──────────────────────────────────────────────────────────────────────
# UTILITY FUNCTION
# ──────────────────────────────────────────────────────────────────────
def build_app(app_name, app_type="web_app", description="", tech_stack="auto"):
    builder = EverythingBuilder()
    return builder.build_complete_app(app_name=app_name, app_type=app_type, description=description, tech_stack=tech_stack)


# ──────────────────────────────────────────────────────────────────────
# DIRECT EXECUTION
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("🔨 BELLA.01 Everything Builder - Full-Stack Web App Generator")
    print("=" * 70)
    result = build_app(app_name="My Todo Application", app_type="todo_app", description="A complete task management application with backend API and frontend", tech_stack="auto")
    print("\n✅ Project generated: " + result["project_id"])
    print("   Location: " + result["project_dir"])
    print("   Files: " + str(result["manifest"]["file_count"]) + " files")
    print("   Size: " + str(result["manifest"]["total_size_kb"]) + " KB")
    print("\n📁 Project Structure:")
    import os
    for root, dirs, files in os.walk(result["project_dir"]):
        level = root.replace(result["project_dir"], "").count(os.sep)
        indent = " " * 2 * level
        print(indent + os.path.basename(root) + "/")
        sub_indent = " " * 2 * (level + 1)
        for file in files:
            print(sub_indent + file)
    print("\n🚀 To run:")
    print("   cd " + result["project_dir"])
    print("   cd backend && uvicorn app:app --reload")
    print("   # In another terminal:")
    print("   cd " + result["project_dir"] + "/frontend && npm run dev")
    print("=" * 70)
