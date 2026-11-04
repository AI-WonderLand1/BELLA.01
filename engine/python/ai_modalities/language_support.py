"""Multi-language support for AI-generated code and applications.

Supports translation of generated code, natural language prompts,
and interface text into multiple programming languages and human languages.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────────
# PROGRAMMING LANGUAGE TRANSLATION
# ────────────────────────────────────────────────────────────────────

PYTHON_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "class", "import",
    "from", "as", "try", "except", "finally", "return", "yield",
    "with", "as", "raise", "break", "continue", "in", "is", "not",
    "and", "or", "True", "False", "None", "lambda", "global", "nonlocal"
}

JAVASCRIPT_KEYWORDS = {
    "if", "else", "else if", "for", "while", "function", "const", "let",
    "var", "try", "catch", "finally", "return", "yield", "void",
    "break", "continue", "in", "instanceof", "new", "this", "typeof",
    "return", "delete", "debugger", "export", "import", "default"
}

TYPESCRIPT_KEYWORDS = {
    "if", "else", "elseif", "for", "while", "function", "const", "let",
    "var", "try", "catch", "finally", "return", "yield", "void",
    "break", "continue", "in", "instanceof", "new", "this", "type",
    "interface", "implements", "export", "import", "extends", "type"
}

# ────────────────────────────────────────────────────────────────────
# PROGRAMMING LANGUAGE TRANSLATOR
# ──────────────────────────────────────────────────────────────────


class LanguageTranslator:
    """Translate code between programming languages."""

    @staticmethod
    def translate_python_to_js(code: str) -> str:
        """Translate Python code to JavaScript/TypeScript."""
        lines = code.split("\n")
        js_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Handle def -> function
            match = re.match(r"^\s*def\s+(\w+)(?:\(([^)]*)\))?\s*(?:->\s*(.*))?\s*:", line)
            if match:
                func_name = match.group(1)
                params = match.group(2) or ""
                returns = match.group(3) or ""

                # Convert params
                param_str = params
                if "self" in param_str:
                    param_str = param_str.replace("self", "")

                # Handle return
                ret_str = ""
                if returns:
                    ret_str = f": {returns}"

                js_lines.append(f"function {func_name}({param_str}) {{")
                js_lines.append(f"  // {returns or 'return value'}")

                # Increase indentation for body
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    body_line = lines[i].lstrip()
                    # Skip 'return' at top level of function translation
                    if body_line.startswith("return "):
                        js_lines.append(f"  return {body_line[len('return '):]}")
                    elif body_line.startswith("print("):
                        js_lines.append(f"  console.log{body_line[len('print('):]}")
                    elif body_line.startswith("print"):
                        js_lines.append(f"  console.log{body_line[len('print'):]}")
                    else:
                        js_lines.append(f"  {body_line}")
                    i += 1

                # Close function
                js_lines.append("}")
                continue

            # Handle print -> console.log
            if line.strip().startswith("print("):
                js_lines.append(f"console.log{line.strip()[6:]}")
                i += 1
                continue

            # Handle import -> require/import
            if line.strip().startswith("import "):
                # Simple handling
                if "from" in line:
                    # from module import something
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        mod = parts[2] if parts[1] == "from" else parts[1]
                        items = parts[-1] if parts[-1].startswith("import") else ""
                        js_lines.append(f"import {{ {items} }} from '{mod}';")
                else:
                    mod = line.strip().replace("import ", "").rstrip(";")
                    js_lines.append(f"import {{ {mod} }} from '{mod}';")
                i += 1
                continue

            # Handle from import
            if line.strip().startswith("from "):
                js_lines.append(f"// {line.strip()}")
                i += 1
                continue

            # Handle if -> if
            if re.match(r"^\s*if\s+", line):
                js_lines.append(line.replace("if ", "if ").replace(":", ""))
                i += 1
                continue

            # Handle for -> for
            if re.match(r"^\s*for\s+(\w+)\s+in\s+", line):
                match = re.match(r"^\s*for\s+(\w+)\s+in\s+(.+):", line)
                if match:
                    var = match.group(1)
                    coll = match.group(2)
                    js_lines.append(f"for (let {var} of {coll}) {{")
                    i += 1
                    # Add indented body lines
                    while i < len(lines):
                        next_line = lines[i]
                        if next_line.startswith("    ") and not any(
                            next_line.strip().startswith(kw) for kw in ["def", "if", "for", "while", "class"]
                        ):
                            js_lines.append(f"  {next_line.strip()}")
                            i += 1
                        else:
                            break
                    js_lines.append("}")
                    continue

            # Handle while -> while
            if re.match(r"^\s*while\s+", line):
                js_lines.append(line.replace("while ", "while "))
                i += 1
                continue

            # Handle return -> return
            if line.strip().startswith("return "):
                js_lines.append(f"  return {line.strip()[7:]}")
                i += 1
                continue

            # Default: pass through
            js_lines.append(line)
            i += 1

        return "\n".join(js_lines)

    @staticmethod
    def translate_js_to_python(code: str) -> str:
        """Translate JavaScript/TypeScript to Python."""
        lines = code.split("\n")
        py_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Handle function -> def
            match = re.match(r"^\s*function\s+(\w+)?\s*\(([^)]*)\)\s*\{", line)
            if match:
                func_name = match.group(1) or "function_name"
                params = match.group(2) or ""

                py_lines.append(f"def {func_name}({params}):")
                i += 1

                # Add indented body
                while i < len(lines):
                    next_line = lines[i]
                    # Stop at closing brace or next function
                    if next_line.strip() == "}":
                        i += 1
                        break
                    if next_line.strip() and not next_line.startswith("  "):
                        # New function or class
                        if re.match(r"^\s*(def |class |import |from )", next_line):
                            break
                        py_lines.append(f"  {next_line.strip()}")
                    i += 1
                continue

            # Handle arrow function -> lambda
            if "=>" in line and "function" not in line:
                # Simplified arrow function handling
                arrow_part = line.split("=>")
                if len(arrow_part) == 2:
                    params = arrow_part[0].strip()
                    body = arrow_part[1].strip()
                    if body.startswith("return "):
                        py_lines.append(f"lambda {params}: {body[len('return '):]}")
                    else:
                        py_lines.append(f"lambda {params}: {body}")
                i += 1
                continue

            # Handle console.log -> print
            if line.strip().startswith("console.log"):
                py_lines.append(f"print{line.strip()[len('console.log'):]}")
                i += 1
                continue

            # Default: pass through
            py_lines.append(line)
            i += 1

        return "\n".join(py_lines)

    @staticmethod
    def translate_code(code: str, from_lang: str = "python", to_lang: str = "javascript") -> str:
        """Translate code between languages."""
        if from_lang == "python" and to_lang == "javascript":
            return LanguageTranslator.translate_python_to_js(code)
        elif from_lang == "javascript" and to_lang == "python":
            return LanguageTranslator.translate_js_to_python(code)
        else:
            # Generic: just annotate
            return f"// Translation from {from_lang} to {to_lang} not fully supported yet\\n// Original: {code}"


# ────────────────────────────────────────────────────────────────────
# HUMAN LANGUAGE TRANSLATION (Text)
# ──────────────────────────────────────────────────────────────────


class HumanLanguageTranslator:
    """Translate interface text between human languages."""

    # Simple bilingual dictionaries for common phrases
    TRANSLATIONS = {
        "en": {
            "welcome": "Welcome",
            "hello": "Hello",
            "goodbye": "Goodbye",
            "submit": "Submit",
            "cancel": "Cancel",
            "save": "Save",
            "delete": "Delete",
            "error": "Error",
            "success": "Success",
            "loading": "Loading",
        },
        "es": {
            "welcome": "Bienvenido",
            "hello": "Hola",
            "goodbye": "Adiós",
            "submit": "Enviar",
            "cancel": "Cancelar",
            "save": "Guardar",
            "delete": "Eliminar",
            "error": "Error",
            "success": "Éxito",
            "loading": "Cargando",
        },
        "fr": {
            "welcome": "Bienvenue",
            "hello": "Bonjour",
            "goodbye": "Au revoir",
            "submit": "Soumettre",
            "cancel": "Annuler",
            "save": "Enregistrer",
            "delete": "Supprimer",
            "error": "Erreur",
            "success": "Succès",
            "loading": "Chargement",
        },
        "de": {
            "welcome": "Willkommen",
            "hello": "Hallo",
            "goodbye": "Tschüss",
            "submit": "Absenden",
            "cancel": "Abbrechen",
            "save": "Speichern",
            "delete": "Löschen",
            "error": "Fehler",
            "success": "Erfolg",
            "loading": "Laden",
        },
    }

    @classmethod
    def translate_text(cls, text: str, target_lang: str = "en", source_lang: str = "en") -> str:
        """Translate text to target language."""
        # Check if we have a direct translation
        translations = cls.TRANSLATIONS.get(target_lang, cls.TRANSLATIONS["en"])
        
        # Try to translate known phrases
        text_lower = text.lower().strip()
        if text_lower in translations:
            return translations[text_lower]
        
        # Fallback: return with language prefix
        if target_lang != "en":
            return f"[{target_lang}] {text}"
        return text

    @classmethod
    def translate_interface(cls, interface_spec: Dict[str, str], target_lang: str = "en") -> Dict[str, str]:
        """Translate entire interface dictionary."""
        translated = {}
        for key, value in interface_spec.items():
            translated[key] = cls.translate_text(value, target_lang=target_lang)
        return translated


# ────────────────────────────────────────────────────────────────────
# MULTI-LANGUAGE APP GENERATOR (STANDALONE - NO INHERITANCE)
# ──────────────────────────────────────────────────────────────────


class MultiLanguageBuilder:
    """Builder that generates apps in multiple languages.

    Composes EverythingBuilder functionality without inheritance issues.
    """
    
    def __init__(self, supported_langs=None, code_translator=None):
        self.supported_langs = supported_langs or ["en", "es", "fr", "de"]
        self.code_translator = code_translator or LanguageTranslator()
        self.project_counter = 0

    def build_app_in_language(
        self,
        app_name: str,
        language: str = "en",
        app_type: str = "web_app",
        description: str = "",
    ) -> Dict[str, Any]:
        """Build an app with interface in specified language."""
        self.project_counter += 1
        project_id = f"app_{self.project_counter}_{app_name.replace(' ', '_')}"
        
        # Build description key
        desc_key = "description"
        
        # Translate the interface texts if needed
        localized = False
        if language != "en":
            # Translate description
            if desc_key in description if isinstance(description, dict) else hasattr(description, desc_key):
                pass
            # Simple translation of the description string
            if description:
                translated_desc = HumanLanguageTranslator.translate_text(
                    description, target_lang=language
                )
            else:
                translated_desc = description
            localized = True
        else:
            translated_desc = description
            
        return {
            "project_id": project_id,
            "language": language,
            "localized": localized,
            "description": translated_desc,
            "app_type": app_type,
            "supported_languages": self.supported_langs
        }

    def generate_code_in_language(
        self,
        code: str,
        target_lang: str = "javascript",
        source_lang: str = "python",
    ) -> str:
        """Generate code in target programming language."""
        return self.code_translator.translate_code(
            code, from_lang=source_lang, to_lang=target_lang
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported human languages."""
        return self.supported_langs

    def get_supported_programming_languages(self) -> Dict[str, str]:
        """Get supported programming language mappings."""
        return {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "jsx": "React JSX",
            "html": "HTML",
            "css": "CSS",
        }


# ────────────────────────────────────────────────────────────────────
# SECURITY FEATURES
# ──────────────────────────────────────────────────────────────────


class SecurityScanner:
    """Security scanning for generated code."""

    @staticmethod
    def scan_for_secrets(code: str) -> List[Dict[str, Any]]:
        """Scan code for potential secret keys/credentials."""
        findings = []

        # Patterns for common secrets
        secret_patterns = [
            (r'(?i)api[key secret]\s*=\s*["\'`][^"\'`]{8,}', "API Key/Secret"),
            (r'(?i)password\s*=\s*["\'`][^"\'`]{4,}', "Password"),
            (r'(?i)token\s*=\s*["\'`][^"\'`]{16,}', "Authentication Token"),
            (r'(?i)sk-[a-zA-Z0-9]{20,}', "OpenAI-style Secret Key"),
            (r'(?i)bearer\s+["\'][^"\']+["\']', "Bearer Token"),
        ]

        for pattern, description in secret_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                findings.append({
                    "type": "potential_secret",
                    "description": description,
                    "line": code[:match.start()].count("\n") + 1,
                    "snippet": match.group(),
                    "severity": "high",
                    "recommendation": "Move secret to environment variable or secret manager",
                })

        return findings

    @staticmethod
    def scan_for_dangerous_code(code: str) -> List[Dict[str, Any]]:
        """Scan code for dangerous/suspicious patterns."""
        findings = []

        dangerous_patterns = [
            (r'(?i)exec\s*\(', "exec() function execution"),
            (r'(?i)eval\s*\(', "eval() function execution"),
            (r'(?i)system\s*\(', "system() command execution"),
            (r'(?i)subprocess\.(call|run|Popen)(\s*\\?)', "subprocess execution"),
            (r'(?i)os\.system\s*\(', "os.system() execution"),
            (r'(?i)os\.popen\s*\(', "os.popen() execution"),
        ]

        for pattern, description in dangerous_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                findings.append({
                    "type": "dangerous_pattern",
                    "description": description,
                    "line": code[:match.start()].count("\n") + 1,
                    "snippet": match.group(),
                    "severity": "critical",
                    "recommendation": "Avoid dynamic code execution; use safe alternatives",
                })

        return findings

    @staticmethod
    def scan_complete(code: str) -> Dict[str, Any]:
        """Run complete security scan on code."""
        return {
            "secrets": SecurityScanner.scan_for_secrets(code),
            "dangerous_patterns": SecurityScanner.scan_for_dangerous_code(code),
            "overall_safe": (
                len(SecurityScanner.scan_for_secrets(code)) == 0
                and len(SecurityScanner.scan_for_dangerous_code(code)) == 0
            ),
            "scan_timestamp": __import__("datetime").datetime.now().isoformat(),
        }


# ────────────────────────────────────────────────────────────────────
# VOICE I/O SUPPORT (Text-to-Speech & Speech-to-Text)
# ──────────────────────────────────────────────────────────────────
# Note: Real TTS/STT would require external libraries (pyttsx3, Whisper, etc.)
# This provides the framework structure.

class VoiceIO:
    """Voice Input/Output framework stub for BELLA.01."""

    @staticmethod
    def text_to_speech(text: str, language: str = "en") -> bytes:
        """Convert text to speech (stub - requires pyttsx3 or similar)."""
        print(f"[TTS] Would speak ({language}): {text[:80]}...")
        return b"# TTS stub - would generate audio bytes here"

    @staticmethod
    def speech_to_text(audio_data: bytes = None, language: str = "en") -> str:
        """Convert speech to text (stub - requires Whisper or similar)."""
        # In production: use openai Whisper API or pywhisper
        # For now, return mock response based on language
        mock_responses = {
            "en": "Hello, how can I help you?",
            "es": "Hola, ¿cómo puedo ayudarte?",
            "fr": "Bonjour, comment puis-je vous aider?",
            "de": "Hallo, wie kann ich helfen?",
        }
        return mock_responses.get(language, mock_responses["en"])

    @staticmethod
    def is_available() -> bool:
        """Check if voice I/O is available."""
        return False


# ────────────────────────────────────────────────────────────────────
# EXPORT CLASSES
# ────────────────────────────────────────────────────────────────────
__all__ = [
    "LanguageTranslator",
    "HumanLanguageTranslator",
    "MultiLanguageBuilder",
    "SecurityScanner",
    "VoiceIO",
]


if __name__ == "__main__":
    print("=" * 60)
    print("🌐 BELLA.01 Language & Security Support Module")
    print("=" * 60)

    # Test translations
    print("\n--- Human Language Translation ---")
    for lang in ["en", "es", "fr", "de"]:
        text = "welcome"
        translated = HumanLanguageTranslator.translate_text(text, target_lang=lang)
        print(f"  {lang}: '{text}' -> '{translated}'")

    # Test code translation
    print("\n--- Code Translation ---")
    python_code = 'def calculate_total(items):\n    total = 0\n    for item in items:\n        total += item.price\n    return total'
    js_code = LanguageTranslator.translate_python_to_js(python_code)
    print(f"  Python -> JavaScript: {js_code[:80]}...")

    # Test security scanning
    print("\n--- Security Scanning ---")
    test_code = 'api_key = "sk-2aBcD3eF4gH5iJ6kL7mN8oP9"\nresult = os.system("echo hello")'
    scan_results = SecurityScanner.scan_complete(test_code)
    print(f"  Secrets found: {len(scan_results['secrets'])}")
    print(f"  Dangerous patterns: {len(scan_results['dangerous_patterns'])}")
    print(f"  Overall safe: {scan_results['overall_safe']}")

    # Test multi-language builder
    print("\n--- Multi-Language Support ---")
    builder = MultiLanguageBuilder()
    supported_langs = builder.get_supported_languages()
    print(f"  Supported human languages: {supported_langs}")

    prog_langs = builder.get_supported_programming_languages()
    print(f"  Supported programming languages: {list(prog_langs.keys())}")

    # Translate code in different languages
    print("\n--- Code Translation Demo ---")
    python_code = 'def hello():\n    print("Hello, World!")'
    js_code = builder.generate_code_in_language(python_code, target_lang="javascript", source_lang="python")
    print(f"  Python -> JavaScript: {js_code[:80]}...")
    
    ts_code = builder.generate_code_in_language(python_code, target_lang="typescript", source_lang="python")
    print(f"  Python -> TypeScript: {ts_code[:80]}...")

    # Voice I/O
    print("\n--- Voice I/O ---")
    print(f"  Available: {VoiceIO.is_available()}")
    VoiceIO.text_to_speech("Hello, this is a test", language="en")

    print("\n" + "=" * 60)
