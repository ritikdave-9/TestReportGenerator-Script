# C# Test Report Step Inserter

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Set environment variable:
   ```powershell
   $env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
   ```

### 3. Configure the Script
Edit the `cs_report_inserter.py` file and update the configuration section:
```python
# Root path to scan for .cs files (modify as needed)
ROOT_PATH = r"C:\path\to\your\cs\files"

# Gemini model configuration  
GEMINI_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"
```

## Usage

### Basic Usage
```bash
python cs_report_inserter.py
```

### Dry Run (Preview Changes)
```bash
python cs_report_inserter.py --dry-run
```

### Custom Root Path
```bash
python cs_report_inserter.py --root-path "C:\path\to\your\project"
```

### Custom API Key
```bash
python cs_report_inserter.py --api-key "your-api-key-here"
```

## What It Does

1. **Scans** all `.cs` files recursively from the root path
2. **Identifies** main flow methods (names containing MainFlow, Flow, or ATP_*Flow)
3. **Extracts** XML `<summary>` comments from all methods in the solution
4. **Finds** method calls within main flow methods
5. **Generates** appropriate StepName and StepDescription using Gemini AI
6. **Inserts** `CoreUserActionsWrappers.Common.ReportStep()` calls before each action

## Example Output

**Before:**
```csharp
public void MainFlowMethod()
{
    ReadCountersInitialState();
    GetCurrentTimeAndInitImpressionPaperCounters();
}
```

**After:**
```csharp
public void MainFlowMethod()
{
    CoreUserActionsWrappers.Common.ReportStep("Read Counters", "Capture initial impression paper counters from the device.");
    ReadCountersInitialState();
    
    CoreUserActionsWrappers.Common.ReportStep("Init Counters", "Initialize impression paper counters and record current timestamp.");
    GetCurrentTimeAndInitImpressionPaperCounters();
}
```

## Logging

The script creates a log file `cs_report_inserter.log` with detailed information about:
- Files scanned and processed
- Methods found and modified
- Any errors or warnings
- Summary of changes made

## Safety Features

- **Dry run mode** to preview changes
- **Backup logging** of all modifications
- **Safe file handling** with proper encoding
- **Error recovery** with fallback generation if Gemini API fails
- **Exclusion rules** to avoid instrumenting system calls

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` environment variable is set
2. **No Files Found**: Check that `ROOT_PATH` points to correct directory
3. **Permission Errors**: Run with appropriate file permissions
4. **Encoding Issues**: Script handles UTF-8 encoding automatically

### Debug Mode
Set logging level to DEBUG for more verbose output:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```