import os
import re

def find_main_flow_methods(content):
    """Find main flow methods that need ReportStep insertions"""
    # Look for public methods that are likely main flow methods
    # They usually have summaries with test flow descriptions
    pattern = r'(/// <summary>.*?/// </summary>.*?public\s+(?:virtual\s+)?void\s+(\w+)\s*\([^)]*\))\s*\{'
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    
    main_flow_methods = []
    for match in matches:
        method_signature = match[0]
        method_name = match[1]
        
        # Check if this looks like a main flow method
        if ('Flow' in method_name or 
            'FullFlow' in method_name or
            'MainFlow' in method_name or
            ('/// Test Flow:' in method_signature or
             '/// test main flow' in method_signature.lower() or
             '+++ test main flow+++' in method_signature.lower())):
            main_flow_methods.append(method_name)
    
    return main_flow_methods

def extract_method_calls(content, method_name):
    """Extract the method calls from a specific method"""
    # Find the method definition
    method_pattern = rf'public\s+(?:virtual\s+)?void\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{([^}}]+(?:\{{[^}}]*\}}[^}}]*)*)\}}'
    match = re.search(method_pattern, content, re.DOTALL)
    
    if not match:
        return []
    
    method_body = match.group(1)
    
    # Find method calls (lines that end with ; and contain method names)
    call_pattern = r'^\s*([A-Z][a-zA-Z0-9_]*\s*\([^)]*\)\s*;)'
    calls = re.findall(call_pattern, method_body, re.MULTILINE)
    
    return calls

# List all files in the Billing directory
billing_dir = r"C:\Users\ritikdave\Documents\ReportGeneratorPythonScript\S4General\S4General\Tests\Billing"
files = [f for f in os.listdir(billing_dir) if f.endswith('.cs')]

print("Files to process:")
for file in files:
    print(f"  - {file}")

print(f"\nFound {len(files)} files to process")