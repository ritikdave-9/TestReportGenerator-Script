#!/usr/bin/env python3
"""
C# Test Report Step Inserter

This script scans C# test files and automatically inserts CoreUserActionsWrappers.Common.ReportStep()
calls before action method calls in main flow methods.

Author: GitHub Copilot
Date: October 28, 2025
"""

import os
import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai

# =============================================================================
# CONFIGURATION
# =============================================================================

# Root path to scan for .cs files (modify as needed)
ROOT_PATH = r"C:\Users\ritikdave\Documents\ReportGeneratorPythonScript\S4General\S4General\Tests\Billing"


# Gemini model configuration
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro"

# Main flow method patterns (case-insensitive)
MAIN_FLOW_PATTERNS = [
    r".*MainFlow.*",
    r".*Flow.*", 
    r"ATP_.*Flow.*",
    r".*FullFlow.*"
]


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance (iterative, memory efficient)."""
    a = a or ""
    b = b or ""
    if len(a) < len(b):
        return _levenshtein(b, a)
    # now len(a) >= len(b)
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _approx_contains_flow(text: str, max_distance: int = 2) -> bool:
    """Return True if the text contains a token approximately matching 'flow'/'fullflow'/'mainflow'.

    This handles small typos like 'FullFllow' or 'FullFlwo'.
    """
    if not text:
        return False
    text_l = text.lower()
    # quick fast-path: exact substring
    if 'flow' in text_l:
        return True

    # break into alpha tokens
    tokens = re.findall(r"[a-zA-Z]+", text_l)
    targets = ['flow', 'fullflow', 'mainflow']
    for tok in tokens:
        for tgt in targets:
            # smaller tokens need smaller tolerance
            tol = 1 if len(tgt) <= 4 else max_distance
            dist = _levenshtein(tok, tgt)
            if dist <= tol:
                return True
            # also check substrings of tok (for cases like 'substrateFlow' where token matches but split joined)
            for i in range(0, max(1, len(tok) - len(tgt) + 1)):
                sub = tok[i:i+len(tgt)]
                if _levenshtein(sub, tgt) <= tol:
                    return True
    return False

# Method call pattern (excludes certain calls we don't want to instrument)
METHOD_CALL_PATTERN = r'^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(\s*(?:[^;{}]*)\s*\)\s*;'

# Excluded method prefixes (don't insert reports for these)
EXCLUDED_PREFIXES = [
    "CoreUserActionsWrappers",
    "Assert",
    "Console",
    "Debug",
    "Log",
    "Thread",
    "Task",
    "await",
    "return",
    "throw",
    "if",
    "else",
    "while",
    "for",
    "foreach",
    "switch",
    "try",
    "catch",
    "finally",
    "using",
    "lock",
    "var",
    "int",
    "string",
    "bool",
    "double",
    "float"
]

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cs_report_inserter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# GEMINI AI CLIENT
# =============================================================================

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove Markdown code fences (```csharp, ```cs, ```), returning plain C#.

        If fenced blocks exist, prefer the block that looks like a C# method
        (has an access modifier and braces). Otherwise, remove fence lines.
        """
        import re as _re
        # Extract fenced blocks if present
        fence_pat = _re.compile(r"```(?:[a-zA-Z]+)?\s*(.*?)```", _re.DOTALL)
        blocks = fence_pat.findall(text)
        if blocks:
            # Prefer block that looks like a C# method
            for blk in blocks:
                if (_re.search(r"\b(public|private|protected|internal)\b", blk)
                        and "{" in blk and "}" in blk and "(" in blk and ")" in blk):
                    return blk.strip()
            # Fallback: longest block
            return max(blocks, key=len).strip()
        # No fenced blocks, just remove any stray ``` lines
        cleaned = _re.sub(r"^```.*$", "", text, flags=_re.MULTILINE).strip()
        return cleaned
        
    def generate_report_step(self, method_name: str, xml_summary: Optional[str] = None) -> Tuple[str, str]:
        """Generate StepName and StepDescription using Gemini AI"""
        
        prompt = f"""
You are helping to generate test reporting steps for C# automation tests.

Given:
- Method name: {method_name}
- XML Summary: {xml_summary if xml_summary else "Not available"}

Please generate:
1. StepName: Convert the method name from PascalCase/camelCase to Title Case with spaces. Keep it very short and high-level.
2. StepDescription: One clear sentence describing what the step does. If XML summary is available and meaningful, rewrite it concisely. If not available or nonsense, infer from method name. Start with capital letter. Must be different from StepName.

Respond ONLY in this exact format:
StepName: [your step name here]
StepDescription: [your step description here]

Examples:
Method: ReadCountersInitialState
XML: Reads the initial counter values from the device
Response:
StepName: Read Counters
StepDescription: Capture initial impression paper counters from the device.

Method: GetCurrentTimeAndInitImpressionPaperCounters  
XML: Not available
Response:
StepName: Init Counters
StepDescription: Initialize impression paper counters and record current timestamp.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.3,
                )
            )
            
            # Parse response
            text = response.text.strip()
            step_name = ""
            step_desc = ""
            
            for line in text.split('\n'):
                if line.startswith('StepName:'):
                    step_name = line.replace('StepName:', '').strip()
                elif line.startswith('StepDescription:'):
                    step_desc = line.replace('StepDescription:', '').strip()
            
            if not step_name or not step_desc:
                raise ValueError("Failed to parse Gemini response")
                
            return step_name, step_desc
            
        except Exception as e:
            logger.warning(f"Gemini API error for method {method_name}: {e}")
            # Fallback: generate from method name
            return self._fallback_generation(method_name)
    
    def _fallback_generation(self, method_name: str) -> Tuple[str, str]:
        """Fallback generation when Gemini fails"""
        # Convert PascalCase to Title Case
        step_name = re.sub(r'([A-Z])', r' \1', method_name).strip()
        step_name = re.sub(r'\s+', ' ', step_name)  # Clean up multiple spaces
        
        # Simple description
        step_desc = f"Execute {step_name.lower()} operation."
        
        return step_name, step_desc

    def generate_full_main_flow(self, original_signature: str, original_method: str, xml_summary: Optional[str] = None) -> str:
        """Ask Gemini to rewrite the complete main-flow method.

        Returns the full method text (signature + body) as a string. On failure,
        returns the original method unchanged.
        """
        prompt = f"""
You are an assistant that rewrites C# main flow methods for automation tests.

Inputs:
- Original method signature and body:
{original_signature}
{original_method}

- XML summary (if available): {xml_summary if xml_summary else 'Not available'}

Requirements:
- Preserve the original method name and signature.
- Insert a reporting call BEFORE each action method call in the method body using exactly:
  CoreUserActionsWrappers.Common.ReportStep("<StepName>", "<StepDescription>");
- StepName rules: derive from the called method name; convert PascalCase/camelCase to Title Case with spaces; keep it very short and high-level.
- StepDescription rules: one clear sentence, not identical to StepName, use or rewrite the XML <summary> when available, otherwise infer from the method name. Start with a capital letter. No numbering.
- Keep original code structure, indentation, and formatting as much as possible.
- Do NOT add any extra comments or explanation.
- Do NOT wrap the output in Markdown code fences (no ``` or ```csharp). Return ONLY the complete C# method (signature and body) as plain text.

Produce the rewritten method now.
"""
        try:
            # Add request timeout and retry logic
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    method_name = original_signature.split('(')[0].strip().split()[-1]  # Extract method name
                    logger.info(f"Requesting Gemini rewrite for {method_name} (attempt {attempt + 1}/{max_retries})...")
                    logger.debug(f"Method size: {len(original_method)} chars")
                    
                    # Use threading for timeout (works on Windows)
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    exception_queue = queue.Queue()
                    
                    def make_request():
                        try:
                            logger.info(f"Starting Gemini API call for {method_name}")
                            response = self.model.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    max_output_tokens=8000,
                                    temperature=0.3,
                                )
                            )
                            logger.info(f"Gemini API call completed for {method_name}")
                            result_queue.put(response.text.strip())
                        except Exception as e:
                            logger.error(f"Gemini API call failed for {method_name}: {e}")
                            exception_queue.put(e)
                    
                    logger.info(f"Creating thread for Gemini request - {method_name}")
                    thread = threading.Thread(target=make_request)
                    thread.daemon = True
                    logger.info(f"Starting thread for Gemini request - {method_name}")
                    thread.start()
                    logger.info(f"Waiting for thread completion (60s timeout) - {method_name}")
                    thread.join(timeout=60)  # 60 second timeout
                    
                    if thread.is_alive():
                        logger.warning(f"Gemini request timed out for {method_name} - thread still alive")
                        raise TimeoutError("Gemini request timed out after 60 seconds")
                    
                    if not exception_queue.empty():
                        raise exception_queue.get()
                    
                    if result_queue.empty():
                        raise ValueError("No response received from Gemini")
                    
                    new_method = result_queue.get()
                    # Ensure we return plain C# without markdown fences
                    new_method = self._strip_code_fences(new_method)
                    if not new_method:
                        raise ValueError("Empty response from Gemini for full method generation")
                    logger.info(f"Gemini rewrite successful on attempt {attempt + 1}")
                    return new_method
                except Exception as retry_e:
                    logger.warning(f"Gemini attempt {attempt + 1} failed: {retry_e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Progressive backoff: 2, 4, 6 seconds
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        raise retry_e
        except Exception as e:
            logger.error(f"Gemini full-method generation failed after all retries for {original_signature}: {e}")
            return original_method

# =============================================================================
# FILE PROCESSING
# =============================================================================

class CSharpProcessor:
    def __init__(self, gemini_client: GeminiClient, dry_run: bool = False, replace_flow: bool = False):
        self.gemini_client = gemini_client
        self.dry_run = dry_run
        self.replace_flow = replace_flow
        self.xml_summaries = {}  # Cache for XML summaries
        self.modified_files = []
        self.all_cs_files: List[str] = []
        self.current_file_path: Optional[str] = None
        
    def scan_files(self, root_path: str) -> List[str]:
        """Scan for all .cs files recursively"""
        cs_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.cs'):
                    cs_files.append(os.path.join(root, file))
        return cs_files
    
    def extract_xml_summaries(self, cs_files: List[str]) -> Dict[str, str]:
        """Extract all XML <summary> comments and map to method names"""
        summaries = {}
        
        xml_pattern = r'<summary>\s*(.*?)\s*</summary>\s*(?:.*?\n)*?\s*(?:public|private|protected|internal).*?(\w+)\s*\('
        
        for file_path in cs_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                matches = re.finditer(xml_pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    summary = match.group(1).strip()
                    method_name = match.group(2)
                    # Clean up summary
                    summary = re.sub(r'\s+', ' ', summary)
                    summaries[method_name] = summary
                    
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
        
        logger.info(f"Extracted {len(summaries)} XML summaries")
        return summaries

    def find_xml_summary(self, method_name: str) -> Optional[str]:
        """Lazy-search for an XML <summary> for a specific method name across files.

        Caches results in self.xml_summaries. This avoids a long upfront scan so the
        script can begin processing files immediately.
        """
        if method_name in self.xml_summaries:
            return self.xml_summaries[method_name]
        import time
        start_time = time.time()
        time_budget_sec = 8  # don't spend more than 8s searching summaries per method

        logger.debug(f"XML summary search start for method '{method_name}' with {len(self.all_cs_files)} files")

        # Compile a pattern that looks for <summary> ... </summary> followed by the method name
        xml_pattern = re.compile(r'<summary>\s*(.*?)\s*</summary>\s*(?:.*?\n)*?\s*(?:public|private|protected|internal).*?\b' + re.escape(method_name) + r'\s*\(', re.DOTALL | re.IGNORECASE)

        # Prefer current file first
        search_files = []
        if self.current_file_path and self.current_file_path in self.all_cs_files:
            search_files.append(self.current_file_path)
        # then the rest
        search_files.extend([p for p in self.all_cs_files if p != self.current_file_path])

        scanned = 0
        for file_path in search_files:
            # Respect time budget
            if time.time() - start_time > time_budget_sec:
                logger.info(f"XML summary search time budget exceeded ({time_budget_sec}s) for '{method_name}'. Skipping further search.")
                break
            try:
                # quick prefilter to skip reading content for files without method name mention
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                scanned += 1
                if method_name not in content:
                    continue
                m = xml_pattern.search(content)
                if m:
                    summary = re.sub(r'\s+', ' ', m.group(1).strip())
                    self.xml_summaries[method_name] = summary
                    logger.debug(f"XML summary found for '{method_name}' in {os.path.basename(file_path)} after scanning {scanned} files")
                    return summary
            except Exception as e:
                logger.debug(f"XML summary scan error in {file_path}: {e}")
                continue

        # Not found; cache negative result
        self.xml_summaries[method_name] = None
        logger.debug(f"XML summary not found for '{method_name}' after scanning {scanned} files")
        return None
    
    def is_main_flow_method(self, method_signature: str) -> bool:
        """Check if method signature matches main flow patterns.

        This now supports fuzzy detection for small typos like 'FullFllow'.
        """
        # Quick regex-based match (existing behavior)
        for pattern in MAIN_FLOW_PATTERNS:
            if re.search(pattern, method_signature, re.IGNORECASE):
                return True

        # Fuzzy token-based match (handles small misspellings)
        try:
            if _approx_contains_flow(method_signature, max_distance=2):
                return True
        except Exception:
            # If something goes wrong in fuzzy matching, don't crash — fall back to regex only
            logger.debug("Fuzzy flow detection failed; falling back to regex-only")

        return False
    
    def should_exclude_method(self, method_call: str) -> bool:
        """Check if method call should be excluded from instrumentation"""
        method_name = method_call.split('(')[0].strip()
        
        for prefix in EXCLUDED_PREFIXES:
            if method_name.lower().startswith(prefix.lower()):
                return True
        
        # Skip if already has ReportStep
        if "ReportStep" in method_call:
            return True
            
        return False
    
    def find_main_flow_methods(self, content: str) -> List[Tuple[int, int, str]]:
        """Find all main flow methods and their line ranges"""
        methods = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for method signature (might span multiple lines)
            if ('public' in line or 'private' in line or 'protected' in line):
                # Collect the full method signature (might span multiple lines)
                full_signature = line
                j = i + 1
                while j < len(lines) and '{' not in lines[j] and '}' not in lines[j]:
                    full_signature += " " + lines[j].strip()
                    j += 1
                
                if ('(' in full_signature and ')' in full_signature and 
                    self.is_main_flow_method(full_signature)):
                    logger.debug(f"Found potential main flow method signature: {full_signature}")
                    
                    method_name = self._extract_method_name(full_signature)
                    if not method_name:
                        i += 1
                        continue
                    
                    # Find opening brace (start from where we left off)
                    brace_line = j - 1
                    while brace_line < len(lines) and '{' not in lines[brace_line]:
                        brace_line += 1
                    
                    if brace_line >= len(lines):
                        i += 1
                        continue
                    
                    # Find closing brace (matching pairs)
                    brace_count = 0
                    end_line = brace_line
                    
                    for k in range(brace_line, len(lines)):
                        brace_count += lines[k].count('{')
                        brace_count -= lines[k].count('}')
                        if brace_count <= 0:
                            end_line = k
                            break
                    
                    methods.append((brace_line + 1, end_line, method_name))
                    logger.info(f"Found main flow method: {method_name} (lines {brace_line + 1}-{end_line})")
                    i = end_line + 1
                else:
                    i += 1
            else:
                i += 1
        
        return methods
    
    def _extract_method_name(self, signature: str) -> Optional[str]:
        """Extract method name from signature"""
        match = re.search(r'\b(\w+)\s*\(', signature)
        return match.group(1) if match else None
    
    def process_method_body(self, lines: List[str], start_line: int, end_line: int) -> List[str]:
        """Process a main flow method body and insert ReportStep calls"""
        modified_lines = lines.copy()
        insertions = []  # List of (line_number, report_step_line) to insert
        
        for i in range(start_line, end_line):
            line = lines[i]
            
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
            
            # Check if this is a method call
            match = re.match(METHOD_CALL_PATTERN, line.strip())
            if match and not self.should_exclude_method(line):
                method_call = match.group(1)
                method_name = method_call.split('.')[-1]  # Get last part after dots
                
                # Get XML summary if available (lazy lookup)
                xml_summary = self.find_xml_summary(method_name)

                # Generate report step using Gemini
                step_name, step_desc = self.gemini_client.generate_report_step(method_name, xml_summary)
                
                # Create the ReportStep line with same indentation
                indent = len(line) - len(line.lstrip())
                report_line = ' ' * indent + f'CoreUserActionsWrappers.Common.ReportStep("{step_name}", "{step_desc}");'
                
                insertions.append((i, report_line))
                logger.info(f"Will insert report step before {method_name}: {step_name}")
        
        # Insert in reverse order to maintain line numbers
        for line_num, report_line in reversed(insertions):
            modified_lines.insert(line_num, report_line)
        
        return modified_lines
    
    def process_file(self, file_path: str) -> bool:
        """Process a single C# file"""
        try:
            # Set current file path for prioritized XML search
            self.current_file_path = file_path
            logger.info(f"STEP 1: Reading file {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            original_line_count = len(lines)
            logger.info(f"STEP 2: File read successfully - {original_line_count} lines")
            
            # Find main flow methods
            logger.info(f"STEP 3: Searching for main flow methods in {os.path.basename(file_path)}")
            main_flow_methods = self.find_main_flow_methods(content)
            logger.info(f"STEP 4: Found {len(main_flow_methods)} main flow methods")
            
            if not main_flow_methods:
                logger.info(f"STEP 5: No main flow methods found in {os.path.basename(file_path)} - SKIPPING")
                return False
            
            # If replace_flow mode is enabled, request Gemini to rewrite the whole method
            if self.replace_flow:
                logger.info(f"STEP 5: Replace flow mode enabled - processing {len(main_flow_methods)} methods")
                modified_lines = lines.copy()
                total_replacements = 0
                # iterate in reverse to keep indexes valid when replacing
                for method_idx, (start_line, end_line, method_name) in enumerate(reversed(main_flow_methods), 1):
                    # start_line is the first line after opening brace; signature likely is above it
                    opening_brace_idx = start_line - 1
                    # find signature start by scanning upwards for visibility modifiers and '('
                    sig_start = opening_brace_idx
                    while sig_start >= 0 and not re.search(r'\b(public|private|protected|internal)\b', lines[sig_start]):
                        sig_start -= 1
                    if sig_start < 0:
                        sig_start = 0

                    # The method block to replace covers from sig_start to end_line (inclusive)
                    original_method_text = '\n'.join(lines[sig_start:end_line + 1])
                    original_signature = lines[sig_start].strip()
                    logger.info(f"STEP 5.{method_idx}: Searching XML summary for method {method_name}")
                    xml_summary = self.find_xml_summary(method_name)
                    logger.info(f"STEP 5.{method_idx}: XML summary {'FOUND' if xml_summary else 'NOT FOUND'} for {method_name}")

                    logger.info(f"STEP 6.{method_idx}: Requesting Gemini rewrite for method {method_name} (lines {sig_start + 1}-{end_line + 1})")
                    logger.info(f"STEP 6.{method_idx}: About to call Gemini API for {method_name}")
                    new_method_text = self.gemini_client.generate_full_main_flow(original_signature, original_method_text, xml_summary)
                    logger.info(f"STEP 6.{method_idx}: Gemini API call completed for {method_name}")

                    if not new_method_text or new_method_text.strip() == original_method_text.strip():
                        logger.info(f"No change returned for {method_name}; skipping replacement")
                        continue

                    new_method_lines = new_method_text.split('\n')
                    # Replace the slice
                    modified_lines[sig_start:end_line + 1] = new_method_lines
                    total_replacements += 1
                    logger.info(f"Replaced method {method_name} in {file_path}")

                if total_replacements > 0:
                    if not self.dry_run:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(modified_lines))
                        logger.info(f"Modified {file_path}: {total_replacements} methods replaced")
                        self.modified_files.append(file_path)
                    else:
                        logger.info(f"[DRY RUN] Would replace {total_replacements} methods in {file_path}")
                    return True

                return False

            # Default behavior: insert individual report steps before calls
            # Process each method (in reverse order to maintain line numbers)
            modified_lines = lines.copy()
            total_insertions = 0
            
            for start_line, end_line, method_name in reversed(main_flow_methods):
                processed_lines = self.process_method_body(modified_lines, start_line, end_line + total_insertions)
                insertions_count = len(processed_lines) - len(modified_lines)
                total_insertions += insertions_count
                modified_lines = processed_lines
                
                logger.info(f"Processed method {method_name}: {insertions_count} report steps added")
            
            if total_insertions > 0:
                if not self.dry_run:
                    # Write back to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(modified_lines))
                    
                    logger.info(f"Modified {file_path}: {total_insertions} report steps added")
                    self.modified_files.append(file_path)
                else:
                    logger.info(f"[DRY RUN] Would modify {file_path}: {total_insertions} report steps")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def run(self, root_path: str):
        """Main processing routine"""
        logger.info(f"Starting C# report step insertion - Root: {root_path}")
        logger.info(f"Dry run mode: {self.dry_run}")
        
        # Scan for files
        cs_files = self.scan_files(root_path)
        logger.info(f"Found {len(cs_files)} C# files")

        if not cs_files:
            logger.warning("No C# files found!")
            return

        # Save file list for lazy XML summary lookups
        self.all_cs_files = cs_files

        # NOTE: We avoid extracting all XML summaries up-front to allow the script to
        # begin modifying files immediately. Summaries are found lazily per-method.

        # Process each file one-by-one so changes are written immediately and visible
        processed_count = 0
        total = len(cs_files)
        for idx, file_path in enumerate(cs_files, start=1):
            logger.info(f"===== STARTING PROCESSING FILE {idx}/{total}: {os.path.basename(file_path)} =====")
            logger.info(f"Full file path: {file_path}")
            try:
                modified = self.process_file(file_path)
                if modified:
                    processed_count += 1
                    logger.info(f"File {idx} SUCCESSFULLY MODIFIED: {os.path.basename(file_path)}")
                else:
                    logger.info(f"File {idx} NOT MODIFIED (no main flow methods): {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"ERROR processing file {idx}: {e}")
                logger.error(f"File path: {file_path}")
                continue
            
            # flush handlers so output appears promptly
            for h in logger.handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            
            # Add small delay between files to avoid API rate limits
            if idx < total:  # Don't delay after the last file
                import time
                logger.info(f"Sleeping 1 second before next file...")
                time.sleep(1)  # 1 second delay between files
                logger.info(f"===== FINISHED PROCESSING FILE {idx}/{total} =====")

        logger.info(f"===== ALL FILES PROCESSING COMPLETED =====")
        logger.info(f"Processing complete: {processed_count} files modified")
        if self.modified_files:
            logger.info("Modified files:")
            for file_path in self.modified_files:
                logger.info(f"  - {file_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Insert report steps into C# test files')
    parser.add_argument('--root-path', default=ROOT_PATH, 
                       help='Root directory to scan for .cs files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying files')
    parser.add_argument('--replace-flow', action='store_true',
                       help='Ask Gemini to rewrite entire main-flow methods and replace them in-place')
    parser.add_argument('--api-key', 
                       help='Gemini API key (overrides GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("Gemini API key not provided. Set GEMINI_API_KEY environment variable or use --api-key")
        return 1
    
    try:
        # Initialize Gemini client
        gemini_client = GeminiClient(api_key)

        # Initialize processor
        processor = CSharpProcessor(gemini_client, dry_run=args.dry_run, replace_flow=args.replace_flow)

        # Run processing
        processor.run(args.root_path)
        
        logger.info("Script completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
