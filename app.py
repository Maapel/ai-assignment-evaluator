from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import threading
import zipfile
import os
import tempfile
import shutil
import ast
import difflib
import subprocess
import time
from typing import Dict, List, Any
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import re
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:8000", "http://localhost:8000"])

# In-memory database for jobs
JOBS_DB: Dict[str, Dict[str, Any]] = {}

# Configure Groq API
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        model = "llama-3.1-8b-instant"  # Higher rate limits: 14.4K RPD, 500K TPD
        print(f"Using Groq model: {model}")
    except Exception as e:
        groq_client = None
        model = None
        print(f"Warning: Failed to configure Groq API: {e}")
else:
    groq_client = None
    model = None
    print("Warning: GROQ_API_KEY not found. AI analysis will be disabled.")

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

def run_evaluation(job_id: str, file_paths: List[str]):
    """Main evaluation function that runs in background thread"""
    try:
        # Update status to processing
        JOBS_DB[job_id]["status"] = "processing"
        JOBS_DB[job_id]["status_message"] = "Starting evaluation..."

        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            student_reports = []

            # Process each submission
            for file_path in file_paths:
                JOBS_DB[job_id]["status_message"] = f"Processing {os.path.basename(file_path)}..."

                # Extract student ID and assignment from filename
                filename = os.path.basename(file_path)
                student_id = filename.split('_')[0]  # e.g., EE24B032

                # Create student temp directory
                student_temp_dir = os.path.join(temp_dir, student_id)
                os.makedirs(student_temp_dir, exist_ok=True)

                # Extract zip file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(student_temp_dir)

                # Find Python files and PDF files
                python_files = []
                pdf_files = []
                for root, dirs, files in os.walk(student_temp_dir):
                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(os.path.join(root, file))
                        elif file.endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))

                if not python_files:
                    student_reports.append({
                        "student_id": student_id,
                        "filename": filename,
                        "functional_score": 0,
                        "report_score": 0,
                        "plagiarism_score": 0,
                        "ai_feedback": "No Python files found in submission",
                        "error": "No Python files found"
                    })
                    continue

                # For simplicity, take the first Python file found
                code_file = python_files[0]

                # Read the code
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code_content = f.read()

                # Extract report text if PDF exists
                report_text = ""
                if pdf_files:
                    report_text = extract_pdf_text(pdf_files[0])

                # Functional testing with Docker
                functional_score = run_docker_tests(code_file, student_temp_dir)

                # Report analysis (simple heuristic based on content)
                report_score = analyze_report_quality(report_text)

                # Plagiarism detection
                plagiarism_score = check_plagiarism(code_content, [r['code'] for r in student_reports if 'code' in r])

                # AI analysis
                ai_feedback = analyze_with_groq(code_content, report_text) if groq_client and model else "AI analysis disabled - no API key"

                # Store code for plagiarism checking against future submissions
                student_reports.append({
                    "student_id": student_id,
                    "filename": filename,
                    "code": code_content,
                    "report_text": report_text,
                    "functional_score": functional_score,
                    "report_score": report_score,
                    "plagiarism_score": plagiarism_score,
                    "ai_feedback": ai_feedback
                })

            # Calculate summary statistics
            total_submissions = len(student_reports)
            avg_functional = sum(r.get('functional_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
            avg_report = sum(r.get('report_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
            avg_plagiarism = sum(r.get('plagiarism_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0

            # Remove code content from final report (not needed in response)
            for report in student_reports:
                report.pop('code', None)
                report.pop('report_text', None)

            final_report = {
                "summary": {
                    "total_submissions": total_submissions,
                    "average_functional_score": round(avg_functional, 2),
                    "average_report_score": round(avg_report, 2),
                    "average_plagiarism_score": round(avg_plagiarism, 2),
                    "generated_at": time.time()
                },
                "student_reports": student_reports
            }

            # Mark job as complete
            JOBS_DB[job_id]["status"] = "complete"
            JOBS_DB[job_id]["status_message"] = "Evaluation complete"
            JOBS_DB[job_id]["report"] = final_report

    except Exception as e:
        # Mark job as failed
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["status_message"] = f"Evaluation failed: {str(e)}"
        print(f"Evaluation error for job {job_id}: {e}")

def run_batch_evaluation(job_id: str):
    """Process a batch job that has already been scanned"""
    try:
        job = JOBS_DB.get(job_id)
        if not job or job.get('status') != 'processing':
            return

        submissions = job.get('submissions', [])
        if not submissions:
            JOBS_DB[job_id]["status"] = "failed"
            JOBS_DB[job_id]["status_message"] = "No submissions found"
            return

        JOBS_DB[job_id]["status_message"] = f"Step 1/4: Found {len(submissions)} submissions. Starting individual processing..."

        # Process each student submission
        student_reports = []
        processed_submissions = []

        for i, submission in enumerate(submissions):
            student_id = submission['student_id']
            filename = submission['filename']
            file_path = submission.get('path', '')

            JOBS_DB[job_id]["status_message"] = f"Step 2/4: Processing {i+1}/{len(submissions)} - {filename}"

            try:
                # Create temporary directory for this student
                student_temp_dir = tempfile.mkdtemp()

                # Extract the student's zip file
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Extracting submission..."
                with zipfile.ZipFile(file_path, 'r') as student_zip:
                    student_zip.extractall(student_temp_dir)

                # Find Python files and PDF files
                python_files = []
                pdf_files = []
                for root, dirs, files in os.walk(student_temp_dir):
                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(os.path.join(root, file))
                        elif file.endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))

                if not python_files:
                    # Clean up temp directory
                    shutil.rmtree(student_temp_dir)
                    error_report = {
                        "student_id": student_id,
                        "filename": filename,
                        "functional_score": 0,
                        "code_quality_score": 0,
                        "report_score": 0,
                        "plagiarism_score": 0,
                        "ai_feedback": "No Python files found in submission",
                        "error": "No Python files found"
                    }
                    student_reports.append(error_report)
                    processed_submissions.append(error_report)
                    JOBS_DB[job_id]["processed_submissions"] = processed_submissions
                    JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: No Python files found"
                    continue

                # Take the first Python file found
                code_file = python_files[0]

                # Read the code
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Analyzing code..."
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code_content = f.read()

                # Extract report text if PDF exists
                report_text = ""
                if pdf_files:
                    JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Extracting report..."
                    report_text = extract_pdf_text(pdf_files[0])

                # Functional testing - use static analysis
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Running functional tests..."
                functional_score = run_static_tests(code_content)

                # LLM-based quality scoring
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Analyzing code quality..."
                code_quality_score = get_llm_code_quality_score(code_content)

                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Analyzing report quality..."
                report_quality_score = get_llm_report_quality_score(report_text) if report_text else 0.0

                # Plagiarism detection
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Checking plagiarism..."
                plagiarism_result = check_plagiarism(
                    code_content,
                    [r['code'] for r in student_reports if 'code' in r],
                    [r['student_id'] for r in student_reports if 'student_id' in r]
                )

                # AI analysis
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Generating AI feedback..."
                ai_feedback = analyze_with_groq(code_content, report_text) if groq_client and model else "AI analysis disabled - no API key"

                # Clean up temp directory
                shutil.rmtree(student_temp_dir)

                # Store results
                student_report = {
                    "student_id": student_id,
                    "filename": filename,
                    "code": code_content,
                    "report_text": report_text,
                    "functional_score": functional_score,
                    "code_quality_score": code_quality_score,
                    "report_score": report_quality_score,
                    "plagiarism_score": plagiarism_result["score"],
                    "plagiarism_details": plagiarism_result["similar_submissions"],
                    "ai_feedback": ai_feedback
                }

                student_reports.append(student_report)
                processed_submissions.append(student_report)

                # Update job with processed submissions
                JOBS_DB[job_id]["processed_submissions"] = processed_submissions

                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Complete"

            except Exception as e:
                # Clean up temp directory if it exists
                try:
                    shutil.rmtree(student_temp_dir)
                except:
                    pass

                # Handle individual student processing errors
                error_report = {
                    "student_id": student_id,
                    "filename": filename,
                    "functional_score": 0,
                    "code_quality_score": 0,
                    "report_score": 0,
                    "plagiarism_score": 0,
                    "ai_feedback": f"Processing failed: {str(e)}",
                    "error": f"Failed to process submission: {str(e)}"
                }
                student_reports.append(error_report)
                processed_submissions.append(error_report)
                JOBS_DB[job_id]["processed_submissions"] = processed_submissions
                JOBS_DB[job_id]["status_message"] = f"Step 3/4: Processing {i+1}/{len(submissions)} - {filename}: Error - {str(e)}"
                continue

        # Calculate summary statistics
        total_submissions = len(student_reports)
        avg_functional = sum(r.get('functional_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
        avg_code_quality = sum(r.get('code_quality_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
        avg_report = sum(r.get('report_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
        avg_plagiarism = sum(r.get('plagiarism_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0

        # Remove code content from final report
        for report in student_reports:
            report.pop('code', None)
            report.pop('report_text', None)

        final_report = {
            "summary": {
                "total_submissions": total_submissions,
                "average_functional_score": round(avg_functional, 2),
                "average_code_quality_score": round(avg_code_quality, 2),
                "average_report_score": round(avg_report, 2),
                "average_plagiarism_score": round(avg_plagiarism, 2),
                "generated_at": time.time()
            },
            "student_reports": student_reports
        }

        # Mark job as complete
        JOBS_DB[job_id]["status"] = "complete"
        JOBS_DB[job_id]["status_message"] = f"Batch evaluation complete! Processed {total_submissions} submissions."
        JOBS_DB[job_id]["report"] = final_report

    except Exception as e:
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["status_message"] = f"Batch evaluation failed: {str(e)}"
        print(f"Batch evaluation error for job {job_id}: {e}")

def get_llm_code_quality_score(code: str) -> float:
    """Get code quality score from LLM"""
    if not groq_client or not model:
        return 0.5  # Default score

    try:
        prompt = f"""
Analyze the code quality of this Python code and provide a score from 0.0 to 1.0.

Code to analyze:
{code[:1000]}

Consider:
- Code structure and organization
- Variable naming and readability
- Function design and documentation
- Best practices adherence
- Potential bugs or issues

Return only a number between 0.0 and 1.0 representing the code quality score.
"""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=10
        )

        response = chat_completion.choices[0].message.content.strip()
        # Extract number from response
        import re
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
        return 0.5

    except Exception as e:
        print(f"LLM code quality scoring error: {e}")
        return 0.5

def get_llm_report_quality_score(report_text: str) -> float:
    """Get report quality score from LLM"""
    if not groq_client or not model or not report_text:
        return 0.5

    try:
        prompt = f"""
Analyze the quality of this assignment report and provide a score from 0.0 to 1.0.

Report to analyze:
{report_text[:1000]}

Consider:
- Structure and organization
- Technical content accuracy
- Clarity and readability
- Completeness of analysis
- Professional presentation

Return only a number between 0.0 and 1.0 representing the report quality score.
"""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=10
        )

        response = chat_completion.choices[0].message.content.strip()
        # Extract number from response
        import re
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
        return 0.5

    except Exception as e:
        print(f"LLM report quality scoring error: {e}")
        return 0.5

def run_static_tests(code_content: str) -> float:
    """Run static analysis tests (faster than Docker for batch processing)"""
    try:
        score = 0.0

        # Check for required functions (keyboard optimization assignment)
        required_functions = ['preprocess_text', 'path_length_cost', 'simulated_annealing']
        implemented_functions = []

        for func_name in required_functions:
            if func_name in code_content:
                implemented_functions.append(func_name)

        score += 0.3 * (len(implemented_functions) / len(required_functions))

        # Check for proper function definitions
        function_definition_score = 0.0
        for func_name in implemented_functions:
            if f"def {func_name}(" in code_content:
                function_definition_score += 1.0 / len(required_functions)
        score += 0.2 * function_definition_score

        # Check for algorithm components
        algorithm_score = 0.0
        sa_indicators = ['simulated_annealing', 'temperature', 'alpha', 'cost', 'swap']
        sa_matches = sum(1 for indicator in sa_indicators if indicator in code_content.lower())
        algorithm_score += 0.2 * (sa_matches / len(sa_indicators))

        path_indicators = ['math.dist', 'math.sqrt', 'distance', 'euclidean']
        path_matches = sum(1 for indicator in path_indicators if indicator in code_content.lower())
        algorithm_score += 0.1 * (path_matches / len(path_indicators))

        preprocess_indicators = ['lower()', 'replace', 'filter', 'allowed']
        preprocess_matches = sum(1 for indicator in preprocess_indicators if indicator in code_content.lower())
        algorithm_score += 0.1 * (preprocess_matches / len(preprocess_indicators))

        score += algorithm_score

        # Check for structure and documentation
        structure_score = 0.0
        if '"""' in code_content or "'''" in code_content:
            structure_score += 0.05
        if 'import' in code_content and ('math' in code_content or 'random' in code_content):
            structure_score += 0.05
        score += structure_score

        # Check for syntax errors
        try:
            ast.parse(code_content)
            score += 0.1
        except SyntaxError:
            score = max(score - 0.2, 0.0)

        return min(max(score, 0.0), 1.0)

    except Exception as e:
        print(f"Static test error: {e}")
        return 0.0

def analyze_report_quality(report_text: str) -> float:
    """Analyze report quality based on content"""
    if not report_text:
        return 0.0

    score = 0.0

    # Check for basic structure indicators
    structure_indicators = [
        'introduction', 'method', 'result', 'conclusion', 'analysis',
        'implementation', 'algorithm', 'performance', 'discussion'
    ]

    text_lower = report_text.lower()
    matches = sum(1 for indicator in structure_indicators if indicator in text_lower)
    score += 0.4 * (matches / len(structure_indicators))

    # Check for technical content
    technical_indicators = [
        'python', 'code', 'function', 'algorithm', 'optimization',
        'simulated annealing', 'keyboard', 'layout', 'cost'
    ]

    tech_matches = sum(1 for indicator in technical_indicators if indicator in text_lower)
    score += 0.3 * (tech_matches / len(technical_indicators))

    # Check for length and detail
    word_count = len(report_text.split())
    if word_count > 100:
        score += 0.2
    elif word_count > 50:
        score += 0.1

    # Check for proper formatting
    if len(report_text) > 500:  # Reasonable length
        score += 0.1

    return min(max(score, 0.0), 1.0)

def run_docker_tests(code_file: str, student_dir: str) -> float:
    """Run functional tests using Docker"""
    try:
        # Check if this is a keyboard optimization assignment
        with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()

        is_keyboard_assignment = any(keyword in code_content.lower() for keyword in [
            'keyboard', 'layout', 'qwerty', 'simulated annealing', 'path_length_cost'
        ])

        if is_keyboard_assignment:
            return run_keyboard_optimization_tests(code_file, student_dir, code_content)
        else:
            return run_general_tests(code_file, student_dir)

    except Exception as e:
        print(f"Docker test error: {e}")
        return 0.0

def run_keyboard_optimization_tests(code_file: str, student_dir: str, code_content: str) -> float:
    """Specific tests for keyboard layout optimization assignment using static analysis"""
    try:
        score = 0.0

        # Test 1: Check if required functions exist (0.3 points)
        required_functions = ['preprocess_text', 'path_length_cost', 'simulated_annealing']
        implemented_functions = []

        for func_name in required_functions:
            if func_name in code_content:
                implemented_functions.append(func_name)

        score += 0.3 * (len(implemented_functions) / len(required_functions))

        # Test 2: Check for proper function definitions (0.2 points)
        function_definition_score = 0.0
        for func_name in implemented_functions:
            # Look for function definition pattern
            if f"def {func_name}(" in code_content:
                function_definition_score += 1.0 / len(required_functions)
        score += 0.2 * function_definition_score

        # Test 3: Check for key algorithm components (0.4 points)
        algorithm_score = 0.0

        # Check for simulated annealing components
        sa_indicators = ['simulated_annealing', 'temperature', 'alpha', 'cost', 'swap']
        sa_matches = sum(1 for indicator in sa_indicators if indicator in code_content.lower())
        algorithm_score += 0.2 * (sa_matches / len(sa_indicators))

        # Check for path length calculation
        path_indicators = ['math.dist', 'math.sqrt', 'distance', 'euclidean']
        path_matches = sum(1 for indicator in path_indicators if indicator in code_content.lower())
        algorithm_score += 0.1 * (path_matches / len(path_indicators))

        # Check for text preprocessing
        preprocess_indicators = ['lower()', 'replace', 'filter', 'allowed']
        preprocess_matches = sum(1 for indicator in preprocess_indicators if indicator in code_content.lower())
        algorithm_score += 0.1 * (preprocess_matches / len(preprocess_indicators))

        score += algorithm_score

        # Test 4: Check for proper structure and documentation (0.1 points)
        structure_score = 0.0

        # Check for docstrings or comments
        if '"""' in code_content or "'''" in code_content:
            structure_score += 0.05

        # Check for proper imports
        if 'import' in code_content and ('math' in code_content or 'random' in code_content):
            structure_score += 0.05

        score += structure_score

        # Check for syntax errors by parsing AST
        try:
            ast.parse(code_content)
            score += 0.1  # Bonus for valid syntax
        except SyntaxError:
            score = max(score - 0.2, 0.0)  # Penalty for syntax errors

        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

    except Exception as e:
        print(f"Keyboard optimization static analysis error: {e}")
        return 0.0

def run_general_tests(code_file: str, student_dir: str) -> float:
    """General tests for non-specific assignments"""
    try:
        # Create a simple test script that imports and runs basic checks
        test_script = f"""
import sys
import os
sys.path.insert(0, '/app')

try:
    # Try to import the student's module
    import importlib.util
    spec = importlib.util.spec_from_file_location("student_code", "/app/student_code.py")
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)

    # Basic functionality check - look for common function patterns
    functions = [name for name in dir(student_module) if not name.startswith('_')]
    if functions:
        score = 0.8  # Basic score for having functions
    else:
        score = 0.3  # Low score for no functions

    # Check for syntax errors by parsing AST
    import ast
    with open('/app/student_code.py', 'r') as f:
        code = f.read()
    ast.parse(code)  # This will raise SyntaxError if invalid
    score += 0.2  # Bonus for valid syntax

    print(f"SCORE:{{score}}")

except SyntaxError as e:
    print("SCORE:0.1")  # Very low score for syntax errors
    print(f"ERROR: Syntax error - {{e}}")
except Exception as e:
    print(f"SCORE:0.2")  # Low score for runtime errors
    print(f"ERROR: Runtime error - {{e}}")
"""
        # Write test script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name

        # Copy student code to a file named student_code.py
        student_code_dest = os.path.join(student_dir, 'student_code.py')
        shutil.copy2(code_file, student_code_dest)

        # Run Docker container with the test
        docker_cmd = [
            'sudo', 'docker', 'run', '--rm',
            '-v', f'{student_dir}:/app:ro',
            '-v', f'{test_file}:/test.py:ro',
            'python:3.9-slim',
            'python', '/test.py'
        ]

        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)

        # Clean up
        os.unlink(test_file)

        # Parse the output for score
        output = result.stdout + result.stderr
        if "SCORE:" in output:
            score_line = [line for line in output.split('\n') if line.startswith("SCORE:")][0]
            score = float(score_line.split(":")[1])
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
        else:
            return 0.0

    except Exception as e:
        print(f"General test error: {e}")
        return 0.0

        # Write test script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name

        # Copy student code to a file named student_code.py
        student_code_dest = os.path.join(student_dir, 'student_code.py')
        shutil.copy2(code_file, student_code_dest)

        # Run Docker container with the test
        docker_cmd = [
            'docker', 'run', '--rm',
            '-v', f'{student_dir}:/app:ro',
            '-v', f'{test_file}:/test.py:ro',
            'python:3.9-slim',
            'python', '/test.py'
        ]

        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)

        # Clean up
        os.unlink(test_file)

        # Parse the output for score
        output = result.stdout + result.stderr
        if "SCORE:" in output:
            score_line = [line for line in output.split('\n') if line.startswith("SCORE:")][0]
            score = float(score_line.split(":")[1])
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
        else:
            return 0.0

    except Exception as e:
        print(f"Docker test error: {e}")
        return 0.0

def check_plagiarism(code: str, other_codes: List[str], student_ids: List[str] = None) -> dict:
    """Check for plagiarism using AST and difflib, return detailed similarity info"""
    if not other_codes:
        return {"score": 0.0, "similar_submissions": []}

    try:
        # Parse AST to normalize code structure
        def normalize_code(code_str):
            try:
                tree = ast.parse(code_str)
                # Remove comments and docstrings for comparison
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Expr, ast.Constant)) and isinstance(node.value, str):
                        # Remove docstrings
                        if isinstance(node, ast.Expr) and isinstance(node.value, str):
                            node.value = ""
                    elif hasattr(node, 'name') and node.name:
                        # Keep function/variable names for structure comparison
                        pass
                return ast.unparse(tree) if hasattr(ast, 'unparse') else code_str
            except:
                return code_str

        normalized_current = normalize_code(code)

        similarities = []
        max_similarity = 0.0

        for i, other_code in enumerate(other_codes):
            normalized_other = normalize_code(other_code)

            # Use difflib for similarity scoring
            similarity = difflib.SequenceMatcher(None, normalized_current, normalized_other).ratio()
            max_similarity = max(max_similarity, similarity)

            # Track significant similarities (>30%)
            if similarity > 0.3:
                student_id = student_ids[i] if student_ids and i < len(student_ids) else f"Submission {i+1}"
                similarities.append({
                    "student_id": student_id,
                    "similarity": round(similarity * 100, 1)
                })

        # Sort by similarity (highest first) and take top 3
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_similarities = similarities[:3]

        return {
            "score": round(max_similarity, 2),
            "similar_submissions": top_similarities
        }

    except Exception as e:
        print(f"Plagiarism check error: {e}")
        return {"score": 0.0, "similar_submissions": []}

# Default scoring weights (can be customized by user)
DEFAULT_WEIGHTS = {
    'functional_tests': 0.4,    # 40% - Test cases passed
    'code_quality': 0.4,       # 40% - Code quality (AI assessed)
    'report_quality': 0.2      # 20% - Report quality (AI assessed)
}

def analyze_with_groq(code: str, report_text: str = "", weights: dict = None) -> str:
    """Analyze code quality and report using Groq API with numerical scoring"""
    if not groq_client or not model:
        return "AI analysis disabled"

    if weights is None:
        weights = DEFAULT_WEIGHTS

    try:
        prompt = f"""
You are a computer science educator evaluating a programming assignment. Provide detailed numerical scoring and analysis.

CODE SUBMISSION:
{code[:2000]}

REPORT SUBMISSION (if provided):
{report_text[:1500] if report_text else "No report submitted"}

ASSIGNMENT SCORING WEIGHTS:
- Functional Tests: {weights['functional_tests']*100}% ({weights['functional_tests']*20:.1f}/20 points)
- Code Quality: {weights['code_quality']*100}% ({weights['code_quality']*20:.1f}/20 points)
- Report Quality: {weights['report_quality']*100}% ({weights['report_quality']*20:.1f}/20 points)

Provide analysis in this EXACT structured format:

## 📊 SCORING BREAKDOWN
**Functional Tests:** [X.X/10] (test cases passed - provided by system)
**Code Quality:** [X.X/10] (your AI assessment)
**Report Quality:** [X.X/10] (your AI assessment)
**Weighted Total:** [X.X/20] (calculated using above weights)

## 🔧 CODE QUALITY ANALYSIS (0-10 scale)
**Algorithm Implementation:** [X.X/5.0] - How well core algorithms are implemented
**Code Structure:** [X.X/3.0] - Organization, readability, modularity
**Best Practices:** [X.X/2.0] - PEP8, naming, documentation
**Total Code Score:** [X.X/10]

## 📝 REPORT QUALITY ANALYSIS (0-10 scale)
**Content Quality:** [X.X/4.0] - Accuracy, completeness, analysis depth
**Technical Depth:** [X.X/3.0] - Understanding of concepts, technical details
**Presentation:** [X.X/3.0] - Clarity, structure, professionalism
**Total Report Score:** [X.X/10]

## 🎯 STUDENT FEEDBACK
**Strengths:** 2-3 specific positive points
**Areas for Improvement:** 2-3 specific actionable suggestions
**Learning Objectives:** What concepts the student should focus on next

## ✅ INSTRUCTOR NOTES
[Brief assessment of student's overall understanding and any grading considerations]
"""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        return f"AI analysis failed: {str(e)}"

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Endpoint to submit files for evaluation"""
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({"error": "No files selected"}), 400

        # Generate unique job ID
        job_id = f"job-{uuid.uuid4()}"

        # Save uploaded files temporarily
        temp_file_paths = []
        temp_dir = tempfile.mkdtemp()

        for file in files:
            if file.filename.endswith('.zip'):
                temp_path = os.path.join(temp_dir, file.filename)
                file.save(temp_path)
                temp_file_paths.append(temp_path)

        if not temp_file_paths:
            shutil.rmtree(temp_dir)
            return jsonify({"error": "No valid zip files found"}), 400

        # Create job entry
        JOBS_DB[job_id] = {
            "status": "pending",
            "status_message": "Queued for evaluation...",
            "report": None
        }

        # Start background evaluation
        thread = threading.Thread(target=run_evaluation, args=(job_id, temp_file_paths))
        thread.daemon = True
        thread.start()

        return jsonify({"job_id": job_id}), 202

    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/evaluate/batch', methods=['POST'])
def evaluate_batch():
    """Endpoint to submit a single zip file containing multiple student submissions"""
    try:
        # Check if batch file was uploaded
        if 'batch' not in request.files:
            return jsonify({"error": "No batch file uploaded"}), 400

        batch_file = request.files['batch']
        if not batch_file or batch_file.filename == '':
            return jsonify({"error": "No batch file selected"}), 400

        if not batch_file.filename.endswith('.zip'):
            return jsonify({"error": "Batch file must be a zip archive"}), 400

        # Generate unique job ID
        job_id = f"batch-{uuid.uuid4()}"

        # Save batch file temporarily
        temp_dir = tempfile.mkdtemp()
        batch_path = os.path.join(temp_dir, batch_file.filename)
        batch_file.save(batch_path)

        # Extract and scan submissions first
        try:
            with zipfile.ZipFile(batch_path, 'r') as batch_zip:
                batch_zip.extractall(temp_dir)

            # Find student submissions - handle Moodle assignment structure
            submissions = []

            # Get the immediate contents of the extracted batch directory
            # This avoids scanning temporary directories created by the system
            batch_contents = os.listdir(temp_dir)

            for item in batch_contents:
                item_path = os.path.join(temp_dir, item)

                # Only process directories (student submission folders)
                # Skip temporary directories and system files
                if os.path.isdir(item_path) and not item.startswith('tmp') and not item.startswith('.'):
                    # Look for zip files inside this student folder
                    student_files = os.listdir(item_path)
                    zip_files = [f for f in student_files if f.endswith('.zip')]

                    if zip_files:
                        # Take the first zip file found in the student folder
                        zip_filename = zip_files[0]
                        file_path = os.path.join(item_path, zip_filename)

                        # Validate it's a proper zip file
                        try:
                            with zipfile.ZipFile(file_path, 'r') as test_zip:
                                test_zip.testzip()

                            # Extract student ID from the folder name
                            folder_name = item
                            # Handle different naming patterns
                            if '_' in folder_name:
                                student_id = folder_name.split('_')[0]
                            else:
                                # Try to extract from folder name patterns like "ee24b003 Akshara G_4683_assignsubmission_file_"
                                import re
                                match = re.search(r'(ee24b\d+|[A-Z]+)', folder_name)
                                student_id = match.group(1) if match else folder_name.split()[0]

                            submissions.append({
                                "student_id": student_id,
                                "filename": zip_filename,
                                "folder": folder_name,
                                "path": file_path,
                                "type": "zip_in_folder"
                            })

                        except (zipfile.BadZipFile, zipfile.LargeZipFile):
                            continue

            if not submissions:
                return jsonify({"error": "No valid student submissions found in batch"}), 400

        except Exception as e:
            return jsonify({"error": f"Failed to scan batch file: {str(e)}"}), 400

        # Create job entry with submission list
        JOBS_DB[job_id] = {
            "status": "scanned",
            "status_message": f"Found {len(submissions)} submissions. Ready to process.",
            "submissions": submissions,
            "processed_submissions": [],
            "report": None
        }

        return jsonify({
            "job_id": job_id,
            "submissions": submissions,
            "message": f"Found {len(submissions)} submissions ready for processing"
        }), 200

    except Exception as e:
        return jsonify({"error": f"Batch upload failed: {str(e)}"}), 500

@app.route('/evaluate/batch/start/<job_id>', methods=['POST'])
def start_batch_evaluation(job_id: str):
    """Start the actual batch evaluation process"""
    try:
        job = JOBS_DB.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        if job.get('status') != 'scanned':
            return jsonify({"error": "Job not ready for processing"}), 400

        # Update job status
        job["status"] = "processing"
        job["status_message"] = "Starting batch evaluation..."

        # Start background batch evaluation
        temp_dir = None
        batch_path = None

        # Find the temp directory and batch path from the job data
        # This is a bit hacky but works for this implementation
        for key, value in job.items():
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'filename' in value[0]:
                # This is our submissions list, we can reconstruct paths
                # For now, let's assume the batch file is still in temp
                pass

        # For simplicity, we'll pass the job_id and let the function handle it
        thread = threading.Thread(target=run_batch_evaluation, args=(job_id,))
        thread.daemon = True
        thread.start()

        return jsonify({"message": "Batch evaluation started"}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to start batch evaluation: {str(e)}"}), 500

@app.route('/evaluate/status/<job_id>', methods=['GET'])
def get_status(job_id: str):
    """Endpoint to check evaluation status"""
    job = JOBS_DB.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)

@app.route('/evaluate/results/<job_id>', methods=['GET'])
def get_results(job_id: str):
    """Endpoint to get evaluation results in JSON format"""
    job = JOBS_DB.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job['status'] != 'complete':
        return jsonify({"error": "Evaluation not complete yet"}), 400

    return jsonify(job['report'])

@app.route('/evaluate/download/<job_id>', methods=['GET'])
def download_results(job_id: str):
    """Endpoint to download evaluation results as JSON file"""
    job = JOBS_DB.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job['status'] != 'complete':
        return jsonify({"error": "Evaluation not complete yet"}), 400

    import json
    from flask import Response

    response_data = {
        "job_id": job_id,
        "status": job['status'],
        "status_message": job['status_message'],
        "report": job['report']
    }

    return Response(
        json.dumps(response_data, indent=2),
        mimetype='application/json',
        headers={'Content-disposition': f'attachment; filename=evaluation_{job_id}.json'}
    )

@app.route('/', methods=['GET'])
def index():
    """Serve the main HTML interface"""
    try:
        with open('ai_evaluator.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "HTML file not found", 404

@app.route('/ai_evaluator.html', methods=['GET'])
def serve_html():
    """Serve the HTML interface (for compatibility)"""
    return index()

@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current scoring settings"""
    return jsonify(DEFAULT_WEIGHTS)

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update scoring settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No settings provided"}), 400

        # Validate weights
        required_keys = ['functional_tests', 'code_quality', 'report_quality']
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required weight settings"}), 400

        # Validate weight values (should sum to 1.0 and be between 0 and 1)
        total = sum(data.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            return jsonify({"error": f"Weights must sum to 1.0, got {total}"}), 400

        for key, value in data.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return jsonify({"error": f"Weight {key} must be between 0 and 1"}), 400

        # Update global weights
        global DEFAULT_WEIGHTS
        DEFAULT_WEIGHTS = data.copy()

        return jsonify({"message": "Settings updated successfully", "weights": DEFAULT_WEIGHTS})

    except Exception as e:
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
