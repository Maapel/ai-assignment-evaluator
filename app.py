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
import requests
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:8000", "http://localhost:8000"])

# In-memory database for jobs
JOBS_DB: Dict[str, Dict[str, Any]] = {}

# Configure AI APIs
AI_PROVIDER = os.getenv('AI_PROVIDER', 'openrouter').lower()  # 'groq' or 'openrouter'

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DEFAULT_MODEL = os.getenv('AI_MODEL', 'anthropic/claude-3-haiku:beta')

# Groq Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')

# Initialize API clients
groq_client = None
openrouter_available = False

# Configure Groq API
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq API configured successfully. Model: {GROQ_MODEL}")
    except Exception as e:
        groq_client = None
        print(f"Warning: Failed to configure Groq API: {e}")
else:
    print("Warning: GROQ_API_KEY not found. Groq API will be unavailable.")

# Configure OpenRouter API
if OPENROUTER_API_KEY:
    try:
        # Test the API key
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            openrouter_available = True
            print(f"OpenRouter API configured successfully. Default model: {DEFAULT_MODEL}")
        else:
            print(f"Warning: OpenRouter API test failed: {response.status_code}")
    except Exception as e:
        print(f"Warning: Failed to configure OpenRouter API: {e}")
else:
    print("Warning: OPENROUTER_API_KEY not found. OpenRouter API will be unavailable.")

if not groq_client and not openrouter_available:
    print("Warning: No AI APIs configured. AI analysis will be disabled.")

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

def evaluate_student_submission(student_id: str, filename: str, code_content: str, report_text: str, existing_codes: List[str], existing_student_ids: List[str]):
    """Evaluate a single student submission - shared logic for both individual and batch processing"""

    # Functional testing - use static analysis
    functional_score = run_static_tests(code_content)

    # Plagiarism detection
    plagiarism_result = check_plagiarism(code_content, existing_codes, existing_student_ids)

    # AI analysis and scoring in one call - choose provider based on AI_PROVIDER setting
    if AI_PROVIDER == 'groq' and groq_client:
        ai_result = analyze_with_groq(code_content, report_text, functional_score)
        ai_feedback = ai_result["analysis"]
        code_quality_score = ai_result["code_quality_score"]
        report_quality_score = ai_result["report_quality_score"]
    elif AI_PROVIDER == 'openrouter' and openrouter_available:
        ai_result = analyze_with_openrouter(code_content, report_text, functional_score)
        ai_feedback = ai_result["analysis"]
        code_quality_score = ai_result["code_quality_score"]
        report_quality_score = ai_result["report_quality_score"]
    else:
        ai_feedback = f"AI analysis disabled - {AI_PROVIDER} API not available"
        code_quality_score = 0.5
        report_quality_score = 0.5 if report_text else 0.0

    return {
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

def run_evaluation(job_id: str, file_paths: List[str]):
    """Main evaluation function for individual files that runs in background thread"""
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
                        "code_quality_score": 0,
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

                # Use shared evaluation function
                student_report = evaluate_student_submission(
                    student_id, filename, code_content, report_text,
                    [r['code'] for r in student_reports if 'code' in r],
                    [r['student_id'] for r in student_reports if 'student_id' in r]
                )
                student_reports.append(student_report)

            # Generate final report
            final_report = generate_evaluation_report(student_reports)

            # Mark job as complete
            JOBS_DB[job_id]["status"] = "complete"
            JOBS_DB[job_id]["status_message"] = "Evaluation complete"
            JOBS_DB[job_id]["report"] = final_report

    except Exception as e:
        # Mark job as failed
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["status_message"] = f"Evaluation failed: {str(e)}"
        print(f"Evaluation error for job {job_id}: {e}")

def generate_evaluation_report(student_reports: List[dict]) -> dict:
    """Generate final evaluation report with summary statistics"""
    # Calculate summary statistics
    total_submissions = len(student_reports)
    avg_functional = sum(r.get('functional_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
    avg_code_quality = sum(r.get('code_quality_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0
    avg_report = sum(r.get('report_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0

    # Handle plagiarism score which is now a dict with 'score' key
    avg_plagiarism = sum(r.get('plagiarism_score', {}).get('score', 0) if isinstance(r.get('plagiarism_score'), dict) else r.get('plagiarism_score', 0) for r in student_reports) / total_submissions if total_submissions > 0 else 0

    # Remove code content from final report (not needed in response)
    for report in student_reports:
        report.pop('code', None)
        report.pop('report_text', None)

    return {
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

def run_batch_evaluation(job_id: str):
    """Process a batch job that has already been scanned - Two-pass system for proper plagiarism detection"""
    try:
        job = JOBS_DB.get(job_id)
        if not job or job.get('status') != 'processing':
            return

        submissions = job.get('submissions', [])
        if not submissions:
            JOBS_DB[job_id]["status"] = "failed"
            JOBS_DB[job_id]["status_message"] = "No submissions found"
            return

        JOBS_DB[job_id]["status_message"] = f"Step 1/5: Found {len(submissions)} submissions. Starting extraction..."

        # PASS 1: Extract all submissions and collect code content
        extracted_submissions = []
        all_codes = []
        all_student_ids = []

        for i, submission in enumerate(submissions):
            student_id = submission['student_id']
            filename = submission['filename']
            file_path = submission.get('path', '')

            JOBS_DB[job_id]["status_message"] = f"Step 2/5: Extracting {i+1}/{len(submissions)} - {filename}"

            try:
                # Create temporary directory for this student
                student_temp_dir = tempfile.mkdtemp()

                # Extract the student's zip file
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
                    error_submission = {
                        "student_id": student_id,
                        "filename": filename,
                        "temp_dir": None,
                        "code_content": "",
                        "report_text": "",
                        "has_error": True,
                        "error": "No Python files found"
                    }
                    extracted_submissions.append(error_submission)
                    continue

                # Take the first Python file found
                code_file = python_files[0]

                # Read the code
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code_content = f.read()

                # Extract report text if PDF exists
                report_text = ""
                if pdf_files:
                    report_text = extract_pdf_text(pdf_files[0])

                # Store extracted data
                submission_data = {
                    "student_id": student_id,
                    "filename": filename,
                    "temp_dir": student_temp_dir,
                    "code_content": code_content,
                    "report_text": report_text,
                    "has_error": False,
                    "error": None
                }
                extracted_submissions.append(submission_data)

                # Collect for plagiarism checking
                all_codes.append(code_content)
                all_student_ids.append(student_id)

            except Exception as e:
                # Clean up temp directory if it exists
                try:
                    shutil.rmtree(student_temp_dir)
                except:
                    pass

                error_submission = {
                    "student_id": student_id,
                    "filename": filename,
                    "temp_dir": None,
                    "code_content": "",
                    "report_text": "",
                    "has_error": True,
                    "error": f"Failed to extract submission: {str(e)}"
                }
                extracted_submissions.append(error_submission)

        # PASS 2: Evaluate each submission with full plagiarism context
        JOBS_DB[job_id]["status_message"] = f"Step 3/5: Running evaluations with plagiarism detection..."

        student_reports = []
        processed_submissions = []

        for i, submission_data in enumerate(extracted_submissions):
            student_id = submission_data['student_id']
            filename = submission_data['filename']

            JOBS_DB[job_id]["status_message"] = f"Step 4/5: Evaluating {i+1}/{len(extracted_submissions)} - {filename}"

            if submission_data['has_error']:
                # Handle error case
                error_report = {
                    "student_id": student_id,
                    "filename": filename,
                    "functional_score": 0,
                    "code_quality_score": 0,
                    "report_score": 0,
                    "plagiarism_score": 0,
                    "ai_feedback": f"Processing failed: {submission_data['error']}",
                    "error": submission_data['error']
                }
                student_reports.append(error_report)
                processed_submissions.append(error_report)
                JOBS_DB[job_id]["processed_submissions"] = processed_submissions
                continue

            # Get all other codes for plagiarism checking (exclude current)
            other_codes = [code for j, code in enumerate(all_codes) if j != i]
            other_student_ids = [sid for j, sid in enumerate(all_student_ids) if j != i]

            # Use shared evaluation function with full context
            student_report = evaluate_student_submission(
                student_id, filename, submission_data['code_content'], submission_data['report_text'],
                other_codes, other_student_ids
            )

            # Clean up temp directory
            if submission_data['temp_dir']:
                try:
                    shutil.rmtree(submission_data['temp_dir'])
                except:
                    pass

            student_reports.append(student_report)
            processed_submissions.append(student_report)

            # Update job with processed submissions
            JOBS_DB[job_id]["processed_submissions"] = processed_submissions

        # Generate final report using shared function
        final_report = generate_evaluation_report(student_reports)

        # Mark job as complete
        JOBS_DB[job_id]["status"] = "complete"
        JOBS_DB[job_id]["status_message"] = f"Batch evaluation complete! Processed {len(student_reports)} submissions with comprehensive plagiarism detection."
        JOBS_DB[job_id]["report"] = final_report

    except Exception as e:
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["status_message"] = f"Batch evaluation failed: {str(e)}"
        print(f"Batch evaluation error for job {job_id}: {e}")



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
    """Check for plagiarism using AST and difflib, return detailed similarity info with code snippets"""
    try:
        # Get starter code from global settings
        starter_code = DEFAULT_WEIGHTS.get('starter_code', '').strip()

        # Remove starter code from current code if it exists
        if starter_code:
            code = remove_starter_code(code, starter_code)

        # Remove starter code from all other codes
        filtered_other_codes = []
        filtered_student_ids = []
        for i, other_code in enumerate(other_codes):
            if starter_code:
                filtered_code = remove_starter_code(other_code, starter_code)
            else:
                filtered_code = other_code
            filtered_other_codes.append(filtered_code)
            if student_ids and i < len(student_ids):
                filtered_student_ids.append(student_ids[i])

        # Update references for the rest of the function
        other_codes = filtered_other_codes
        student_ids = filtered_student_ids if student_ids else None

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

        def extract_similar_lines(code1, code2, min_length=3):
            """Extract similar code snippets between two submissions"""
            lines1 = [line.strip() for line in code1.split('\n') if line.strip()]
            lines2 = [line.strip() for line in code2.split('\n') if line.strip()]

            similar_blocks = []
            for i, line1 in enumerate(lines1):
                for j, line2 in enumerate(lines2):
                    if line1 == line2 and len(line1) > 20:  # Only consider substantial lines
                        # Look for consecutive similar lines
                        k = 1
                        while (i + k < len(lines1) and j + k < len(lines2) and
                               lines1[i + k].strip() == lines2[j + k].strip()):
                            k += 1

                        if k >= min_length:  # At least 3 consecutive similar lines
                            block = '\n'.join(lines1[i:i+k])
                            if len(block) > 50:  # Only substantial blocks
                                similar_blocks.append(block[:200] + '...' if len(block) > 200 else block)

            return similar_blocks[:3]  # Return top 3 similar blocks

        def calculate_token_similarity(code1, code2):
            """Calculate similarity based on code tokens (more sophisticated than line-based)"""
            try:
                tree1 = ast.parse(code1)
                tree2 = ast.parse(code2)

                # Extract all names (functions, variables, classes)
                names1 = set()
                names2 = set()

                for node in ast.walk(tree1):
                    if hasattr(node, 'name'):
                        names1.add(node.name)
                    elif hasattr(node, 'id'):
                        names1.add(node.id)

                for node in ast.walk(tree2):
                    if hasattr(node, 'name'):
                        names2.add(node.name)
                    elif hasattr(node, 'id'):
                        names2.add(node.id)

                # Calculate name overlap
                if names1 and names2:
                    name_overlap = len(names1.intersection(names2)) / len(names1.union(names2))
                else:
                    name_overlap = 0.0

                # Calculate structural similarity
                structural_sim = difflib.SequenceMatcher(None,
                    ast.unparse(tree1) if hasattr(ast, 'unparse') else code1,
                    ast.unparse(tree2) if hasattr(ast, 'unparse') else code2
                ).ratio()

                # Weighted combination
                return (name_overlap * 0.3) + (structural_sim * 0.7)

            except:
                # Fallback to simple text similarity
                return difflib.SequenceMatcher(None, code1, code2).ratio()

        normalized_current = normalize_code(code)

        similarities = []
        max_similarity = 0.0

        for i, other_code in enumerate(other_codes):
            normalized_other = normalize_code(other_code)

            # Use enhanced token-based similarity
            similarity = calculate_token_similarity(code, other_code)
            max_similarity = max(max_similarity, similarity)

            # Track significant similarities (>25% for better sensitivity)
            if similarity > 0.25:
                student_id = student_ids[i] if student_ids and i < len(student_ids) else f"Submission {i+1}"

                # Extract similar code snippets
                similar_code_blocks = extract_similar_lines(code, other_code)

                # Determine severity level
                severity = "low"
                if similarity > 0.8:
                    severity = "critical"
                elif similarity > 0.6:
                    severity = "high"
                elif similarity > 0.4:
                    severity = "medium"

                similarities.append({
                    "student_id": student_id,
                    "similarity": round(similarity * 100, 1),
                    "severity": severity,
                    "similar_code_blocks": similar_code_blocks,
                    "reason": f"Code similarity detected: {round(similarity * 100, 1)}% structural match",
                    "analysis": analyze_similarity_type(code, other_code, similarity)
                })

        # Sort by similarity (highest first) and take top 5 (increased from 3)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_similarities = similarities[:5]

        # Determine overall plagiarism risk
        risk_level = "low"
        if max_similarity > 0.8:
            risk_level = "critical"
        elif max_similarity > 0.6:
            risk_level = "high"
        elif max_similarity > 0.4:
            risk_level = "medium"

        return {
            "score": round(max_similarity, 2),
            "risk_level": risk_level,
            "similar_submissions": top_similarities,
            "total_matches": len(similarities)
        }

    except Exception as e:
        print(f"Plagiarism check error: {e}")
        return {"score": 0.0, "risk_level": "unknown", "similar_submissions": [], "total_matches": 0}

def remove_starter_code(code: str, starter_code: str) -> str:
    """Remove starter code from student submission to avoid false plagiarism positives"""
    if not starter_code or not code:
        return code

    try:
        # For keyboard optimization assignment, we want to keep the core algorithm functions
        # but remove boilerplate code that's identical across submissions

        # Parse both codes into AST to understand structure
        try:
            import ast
            code_tree = ast.parse(code)
            starter_tree = ast.parse(starter_code)
        except SyntaxError:
            # If parsing fails, return original code
            return code

        # Extract function and class definitions from both codes
        def extract_definitions(tree):
            functions = {}
            classes = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get the source code for this function
                    functions[node.name] = ast.get_source_segment(code if tree == code_tree else starter_code, node)
                elif isinstance(node, ast.ClassDef):
                    # Get the source code for this class
                    classes[node.name] = ast.get_source_segment(code if tree == code_tree else starter_code, node)
            return functions, classes

        code_functions, code_classes = extract_definitions(code_tree)
        starter_functions, starter_classes = extract_definitions(starter_tree)

        # Functions that are likely to be implemented uniquely by students
        core_functions = {'preprocess_text', 'path_length_cost', 'simulated_annealing'}

        # Classes and functions that are boilerplate and should be filtered
        boilerplate_functions = set()
        boilerplate_classes = set()

        # Check functions
        for func_name in starter_functions:
            if func_name not in core_functions:
                # Check if this function exists in student code and is identical
                if func_name in code_functions:
                    # Compare normalized versions (remove extra whitespace)
                    def normalize_code_segment(code_segment):
                        if not code_segment:
                            return ""
                        lines = [line.rstrip() for line in code_segment.split('\n') if line.strip()]
                        return '\n'.join(lines)

                    starter_normalized = normalize_code_segment(starter_functions[func_name])
                    code_normalized = normalize_code_segment(code_functions[func_name])

                    # If functions are identical, mark as boilerplate
                    if starter_normalized == code_normalized:
                        boilerplate_functions.add(func_name)

        # Check classes - all classes in starter code are likely boilerplate
        for class_name in starter_classes:
            if class_name in code_classes:
                starter_normalized = normalize_code_segment(starter_classes[class_name])
                code_normalized = normalize_code_segment(code_classes[class_name])

                # If classes are identical, mark as boilerplate
                if starter_normalized == code_normalized:
                    boilerplate_classes.add(class_name)

        # Remove boilerplate functions and other common elements
        lines = code.split('\n')
        filtered_lines = []
        skip_until = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip import statements that are in starter code
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Check if this import is in starter code
                if stripped in starter_code:
                    continue

            # Skip type definitions that are identical
            if stripped.startswith(('Point = ', 'Layout = ')):
                if stripped in starter_code:
                    continue

            # Skip function definitions that are boilerplate
            if stripped.startswith('def ') and '(' in stripped:
                func_name = stripped.split('(')[0].replace('def ', '')
                if func_name in boilerplate_functions:
                    # Skip this entire function
                    skip_until = i + 1
                    # Find the end of this function (next function def or end of file)
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line.startswith('def ') or next_line.startswith('class ') or j == len(lines) - 1:
                            skip_until = j
                            break
                    continue

            # Skip class definitions that are boilerplate
            if stripped.startswith('class ') and '(' in stripped:
                class_name = stripped.split('(')[0].replace('class ', '')
                if class_name in boilerplate_classes:
                    # Skip this entire class
                    skip_until = i + 1
                    # Find the end of this class (next class/function def or end of file)
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line.startswith('def ') or next_line.startswith('class ') or j == len(lines) - 1:
                            skip_until = j
                            break
                    continue

            # Handle @dataclass decorated classes
            if stripped.startswith('@dataclass'):
                # Look ahead to find the class definition
                for j in range(i + 1, min(i + 5, len(lines))):  # Look up to 5 lines ahead
                    next_line = lines[j].strip()
                    if next_line.startswith('class ') and '(' in next_line:
                        class_name = next_line.split('(')[0].replace('class ', '')
                        if class_name in boilerplate_classes:
                            # Skip the decorator and entire class
                            skip_until = j + 1
                            # Find the end of this class
                            for k in range(j + 1, len(lines)):
                                end_line = lines[k].strip()
                                if end_line.startswith('def ') or end_line.startswith('class ') or k == len(lines) - 1:
                                    skip_until = k
                                    break
                            break
                if skip_until:
                    continue

            # Skip docstrings and comments that are identical to starter
            if '"""' in line or "'''" in line:
                # Find the matching closing triple quote
                quote_type = '"""' if '"""' in line else "'''"
                start_idx = line.find(quote_type)
                if start_idx != -1:
                    # Extract the docstring
                    docstring_lines = [line[start_idx:]]
                    for j in range(i + 1, len(lines)):
                        docstring_lines.append(lines[j])
                        if quote_type in lines[j]:
                            # Check if this docstring is in starter code
                            docstring_text = '\n'.join(docstring_lines)
                            if docstring_text.strip() in starter_code:
                                # Skip the entire docstring
                                skip_until = j + 1
                            break
                    if skip_until:
                        continue

            # Skip lines that are marked for skipping
            if skip_until and i < skip_until:
                continue
            skip_until = None

            # Skip empty lines that come after skipped content
            if not stripped and skip_until is not None:
                continue

            filtered_lines.append(line)

        # Join back and clean up
        result = '\n'.join(filtered_lines).strip()

        # Remove excessive empty lines
        import re
        result = re.sub(r'\n\n\n+', '\n\n', result)

        # If result is too short, return original code
        if len(result.strip()) < 50:
            return code

        return result

    except Exception as e:
        print(f"Error removing starter code: {e}")
        # Return original code if starter code removal fails
        return code

def analyze_similarity_type(code1, code2, similarity):
    """Analyze the type of similarity detected"""
    try:
        # Check for identical function names
        import re
        func_pattern = r'def\s+(\w+)\s*\('
        funcs1 = set(re.findall(func_pattern, code1))
        funcs2 = set(re.findall(func_pattern, code2))

        common_funcs = funcs1.intersection(funcs2)
        if common_funcs:
            return f"Shared function names: {', '.join(list(common_funcs)[:3])}"

        # Check for identical variable patterns
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        vars1 = set(re.findall(var_pattern, code1))
        vars2 = set(re.findall(var_pattern, code2))

        common_vars = vars1.intersection(vars2)
        if common_vars:
            return f"Shared variable patterns: {', '.join(list(common_vars)[:3])}"

        # Check for algorithmic similarity
        algo_keywords = ['for', 'while', 'if', 'elif', 'else', 'def', 'class', 'import']
        algo_count1 = sum(1 for keyword in algo_keywords if keyword in code1.lower())
        algo_count2 = sum(1 for keyword in algo_keywords if keyword in code2.lower())

        if abs(algo_count1 - algo_count2) <= 2:
            return "Similar algorithmic structure and control flow"

        return "General code structure similarity"

    except:
        return "Code similarity detected"

# Default scoring weights (can be customized by user)
DEFAULT_WEIGHTS = {
    'functional_tests': 0.4,    # 40% - Test cases passed
    'code_quality': 0.4,       # 40% - Code quality (AI assessed)
    'report_quality': 0.2,     # 20% - Report quality (AI assessed)
    'starter_code': ''          # Starter code to exclude from plagiarism detection
}

def analyze_with_groq(code: str, report_text: str = "", functional_score: float = 0.0) -> dict:
    """Generate comprehensive AI analysis and extract quality scores using Groq"""
    if not groq_client:
        return {
            "analysis": "AI analysis disabled - no Groq API key",
            "code_quality_score": 0.5,
            "report_quality_score": 0.5 if report_text else 0.0
        }

    try:
        prompt = f"""
You are a computer science educator providing detailed, justifiable feedback for a programming assignment.

STUDENT CODE:
{code[:1500]}

STUDENT REPORT:
{report_text[:800] if report_text else "No report submitted"}

FUNCTIONAL SCORE: {functional_score * 10:.1f}/10 (test cases passed - provided by system)

Provide a comprehensive analysis following this EXACT TEMPLATE. Replace X.X with numerical scores.

## 🔧 CODE QUALITY ANALYSIS
**Algorithm Implementation:** X.X/5.0 - [2-3 sentence justification]
**Code Structure:** X.X/3.0 - [2-3 sentence justification]
**Best Practices:** X.X/2.0 - [2-3 sentence justification]
**Total Code Score:** X.X/10 (sum of above scores)

## 📝 REPORT QUALITY ANALYSIS
**Content Quality:** X.X/4.0 - [2-3 sentence justification]
**Technical Depth:** X.X/3.0 - [2-3 sentence justification]
**Presentation:** X.X/3.0 - [2-3 sentence justification]
**Total Report Score:** X.X/10 (sum of above scores)

## 🎯 STUDENT FEEDBACK
**Strengths:** [List 2-3 specific positive points with examples]
**Areas for Improvement:** [List 2-3 specific actionable suggestions]
**Learning Objectives:** [List key concepts to focus on next]

## ✅ INSTRUCTOR NOTES
[Detailed assessment of understanding, grading considerations, academic integrity]

CRITICAL: Use EXACT format shown above. Scores must be decimal numbers (e.g., 4.5, 2.0, 1.5). Ensure breakdown scores sum correctly to totals.
"""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            max_tokens=1500
        )

        response = chat_completion.choices[0].message.content.strip()
        print(f"AI Analysis Response Length: {len(response)}")

        # Extract scores from the response
        import re

        # Debug: Print the response to see what we're working with
        print(f"DEBUG: AI Response excerpt:\n{response[:500]}...")

        # Extract code quality breakdown scores
        code_scores = []
        code_patterns = [
            r'\*\*Algorithm Implementation:\*\*\s*(\d+(?:\.\d+)?)/5\.0',
            r'\*\*Code Structure:\*\*\s*(\d+(?:\.\d+)?)/3\.0',
            r'\*\*Best Practices:\*\*\s*(\d+(?:\.\d+)?)/2\.0'
        ]

        for i, pattern in enumerate(code_patterns):
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    code_scores.append(score)
                    print(f"DEBUG: Matched code pattern {i}: {match.group(0)} -> score: {score}")
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Failed to convert code score {i}: {e}")
                    code_scores.append(0.0)
            else:
                print(f"DEBUG: No match for code pattern {i}: {pattern}")
                code_scores.append(0.0)

        # Calculate code quality score (sum of breakdown / 10)
        code_quality_score = sum(code_scores) / 10.0
        code_quality_score = min(max(code_quality_score, 0.0), 1.0)

        # Extract report quality breakdown scores
        report_scores = []
        report_patterns = [
            r'\*\*Content Quality:\*\*\s*(\d+(?:\.\d+)?)/4\.0',
            r'\*\*Technical Depth:\*\*\s*(\d+(?:\.\d+)?)/3\.0',
            r'\*\*Presentation:\*\*\s*(\d+(?:\.\d+)?)/3\.0'
        ]

        for i, pattern in enumerate(report_patterns):
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    report_scores.append(score)
                    print(f"DEBUG: Matched report pattern {i}: {match.group(0)} -> score: {score}")
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Failed to convert report score {i}: {e}")
                    report_scores.append(0.0)
            else:
                print(f"DEBUG: No match for report pattern {i}: {pattern}")
                report_scores.append(0.0)

        # Calculate report quality score (sum of breakdown / 10)
        report_quality_score = sum(report_scores) / 10.0 if report_text else 0.0
        report_quality_score = min(max(report_quality_score, 0.0), 1.0)

        # Add scoring summary at the top
        weights = DEFAULT_WEIGHTS
        functional_display = functional_score * 10
        code_quality_display = code_quality_score * 10
        report_quality_display = report_quality_score * 10
        weighted_total = (functional_score * weights['functional_tests'] +
                         code_quality_score * weights['code_quality'] +
                         report_quality_score * weights['report_quality']) * 20

        scoring_breakdown = f"""## 📊 SCORING BREAKDOWN
**Functional Tests:** {functional_display:.1f}/10 (test cases passed - provided by system)
**Code Quality:** {code_quality_display:.1f}/10 (AI-calculated from breakdown: {code_scores})
**Report Quality:** {report_quality_display:.1f}/10 (AI-calculated from breakdown: {report_scores})
**Weighted Total:** {weighted_total:.1f}/20 (calculated using above weights)

"""

        full_analysis = scoring_breakdown + response

        print(f"Extracted Scores - Code: {code_quality_score} (from {code_scores}), Report: {report_quality_score} (from {report_scores})")

        return {
            "analysis": full_analysis,
            "code_quality_score": code_quality_score,
            "report_quality_score": report_quality_score
        }

    except Exception as e:
        print(f"AI analysis error: {e}")
        return {
            "analysis": f"AI analysis failed: {str(e)}",
            "code_quality_score": 0.5,
            "report_quality_score": 0.5 if report_text else 0.0
        }

def analyze_with_openrouter(code: str, report_text: str = "", functional_score: float = 0.0) -> dict:
    """Generate comprehensive AI analysis and extract quality scores using OpenRouter"""
    if not OPENROUTER_API_KEY:
        return {
            "analysis": "AI analysis disabled - no API key",
            "code_quality_score": 0.5,
            "report_quality_score": 0.5 if report_text else 0.0
        }

    try:
        prompt = f"""
You are a computer science educator providing detailed, justifiable feedback for a programming assignment.

STUDENT CODE:
{code[:1500]}

STUDENT REPORT:
{report_text[:800] if report_text else "No report submitted"}

FUNCTIONAL SCORE: {functional_score * 10:.1f}/10 (test cases passed - provided by system)

Provide a comprehensive analysis following this EXACT TEMPLATE. Replace X.X with numerical scores.

## 🔧 CODE QUALITY ANALYSIS
**Algorithm Implementation:** X.X/5.0 - [2-3 sentence justification]
**Code Structure:** X.X/3.0 - [2-3 sentence justification]
**Best Practices:** X.X/2.0 - [2-3 sentence justification]
**Total Code Score:** X.X/10 (sum of above scores)

## 📝 REPORT QUALITY ANALYSIS
**Content Quality:** X.X/4.0 - [2-3 sentence justification]
**Technical Depth:** X.X/3.0 - [2-3 sentence justification]
**Presentation:** X.X/3.0 - [2-3 sentence justification]
**Total Report Score:** X.X/10 (sum of above scores)

## 🎯 STUDENT FEEDBACK
**Strengths:** [List 2-3 specific positive points with examples]
**Areas for Improvement:** [List 2-3 specific actionable suggestions]
**Learning Objectives:** [List key concepts to focus on next]

## ✅ INSTRUCTOR NOTES
[Detailed assessment of understanding, grading considerations, academic integrity]

CRITICAL: Use EXACT format shown above. Scores must be decimal numbers (e.g., 4.5, 2.0, 1.5). Ensure breakdown scores sum correctly to totals.
"""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.3  # Lower temperature for more consistent scoring
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        ai_response = result['choices'][0]['message']['content'].strip()
        print(f"AI Analysis Response Length: {len(ai_response)}")

        # Extract scores from the response
        import re

        # Debug: Print the response to see what we're working with
        print(f"DEBUG: AI Response excerpt:\n{ai_response[:500]}...")

        # Extract code quality breakdown scores
        code_scores = []
        code_patterns = [
            r'\*\*Algorithm Implementation:\*\*\s*(\d+(?:\.\d+)?)/5\.0',
            r'\*\*Code Structure:\*\*\s*(\d+(?:\.\d+)?)/3\.0',
            r'\*\*Best Practices:\*\*\s*(\d+(?:\.\d+)?)/2\.0'
        ]

        for i, pattern in enumerate(code_patterns):
            match = re.search(pattern, ai_response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    code_scores.append(score)
                    print(f"DEBUG: Matched code pattern {i}: {match.group(0)} -> score: {score}")
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Failed to convert code score {i}: {e}")
                    code_scores.append(0.0)
            else:
                print(f"DEBUG: No match for code pattern {i}: {pattern}")
                code_scores.append(0.0)

        # Calculate code quality score (sum of breakdown / 10)
        code_quality_score = sum(code_scores) / 10.0
        code_quality_score = min(max(code_quality_score, 0.0), 1.0)

        # Extract report quality breakdown scores
        report_scores = []
        report_patterns = [
            r'\*\*Content Quality:\*\*\s*(\d+(?:\.\d+)?)/4\.0',
            r'\*\*Technical Depth:\*\*\s*(\d+(?:\.\d+)?)/3\.0',
            r'\*\*Presentation:\*\*\s*(\d+(?:\.\d+)?)/3\.0'
        ]

        for i, pattern in enumerate(report_patterns):
            match = re.search(pattern, ai_response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    report_scores.append(score)
                    print(f"DEBUG: Matched report pattern {i}: {match.group(0)} -> score: {score}")
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Failed to convert report score {i}: {e}")
                    report_scores.append(0.0)
            else:
                print(f"DEBUG: No match for report pattern {i}: {pattern}")
                report_scores.append(0.0)

        # Calculate report quality score (sum of breakdown / 10)
        report_quality_score = sum(report_scores) / 10.0 if report_text else 0.0
        report_quality_score = min(max(report_quality_score, 0.0), 1.0)

        # Add scoring summary at the top
        weights = DEFAULT_WEIGHTS
        functional_display = functional_score * 10
        code_quality_display = code_quality_score * 10
        report_quality_display = report_quality_score * 10
        weighted_total = (functional_score * weights['functional_tests'] +
                         code_quality_score * weights['code_quality'] +
                         report_quality_score * weights['report_quality']) * 20

        scoring_breakdown = f"""## 📊 SCORING BREAKDOWN
**Functional Tests:** {functional_display:.1f}/10 (test cases passed - provided by system)
**Code Quality:** {code_quality_display:.1f}/10 (AI-calculated from breakdown: {code_scores})
**Report Quality:** {report_quality_display:.1f}/10 (AI-calculated from breakdown: {report_scores})
**Weighted Total:** {weighted_total:.1f}/20 (calculated using above weights)

"""

        full_analysis = scoring_breakdown + ai_response

        print(f"Extracted Scores - Code: {code_quality_score} (from {code_scores}), Report: {report_quality_score} (from {report_scores})")

        return {
            "analysis": full_analysis,
            "code_quality_score": code_quality_score,
            "report_quality_score": report_quality_score
        }

    except Exception as e:
        print(f"AI analysis error: {e}")
        return {
            "analysis": f"AI analysis failed: {str(e)}",
            "code_quality_score": 0.5,
            "report_quality_score": 0.5 if report_text else 0.0
        }

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

@app.route('/playground.html', methods=['GET'])
def serve_playground():
    """Serve the playground page"""
    try:
        with open('playground.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Playground page not found", 404

@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current scoring settings and available models"""
    try:
        # Get available models from OpenRouter
        models = []
        if OPENROUTER_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    # Filter for popular models and sort by context length
                    popular_models = [
                        'anthropic/claude-3-haiku:beta',
                        'anthropic/claude-3-sonnet:beta',
                        'anthropic/claude-3-opus:beta',
                        'openai/gpt-4o-mini',
                        'openai/gpt-4o',
                        'openai/gpt-3.5-turbo',
                        'meta-llama/llama-3.1-8b-instruct',
                        'meta-llama/llama-3.1-70b-instruct',
                        'google/gemini-pro-1.5',
                        'mistralai/mistral-7b-instruct'
                    ]

                    for model in models_data.get('data', []):
                        model_id = model.get('id', '')
                        if model_id in popular_models:
                            models.append({
                                'id': model_id,
                                'name': model.get('name', model_id),
                                'context_length': model.get('context_length', 0)
                            })

                    # Sort by context length (higher first)
                    models.sort(key=lambda x: x['context_length'], reverse=True)

            except Exception as e:
                print(f"Failed to fetch models: {e}")

        settings = DEFAULT_WEIGHTS.copy()
        settings['current_model'] = DEFAULT_MODEL
        settings['available_models'] = models

        return jsonify(settings)

    except Exception as e:
        return jsonify({"error": f"Failed to get settings: {str(e)}"}), 500

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update scoring settings and model selection"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No settings provided"}), 400

        # Validate weights (exclude starter_code from weight validation)
        weight_keys = ['functional_tests', 'code_quality', 'report_quality']
        if not all(key in data for key in weight_keys):
            return jsonify({"error": "Missing required weight settings"}), 400

        # Validate weight values (should sum to 1.0 and be between 0 and 1)
        total = sum(data[key] for key in weight_keys)
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            return jsonify({"error": f"Weights must sum to 1.0, got {total}"}), 400

        for key in weight_keys:
            value = data[key]
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return jsonify({"error": f"Weight {key} must be between 0 and 1"}), 400

        # Validate starter_code (should be string)
        if 'starter_code' in data and not isinstance(data['starter_code'], str):
            return jsonify({"error": "Starter code must be a string"}), 400

        # Validate and update model if provided
        if 'model' in data:
            new_model = data['model']
            if not isinstance(new_model, str) or not new_model.strip():
                return jsonify({"error": "Model must be a non-empty string"}), 400

            # Update the global model variable
            global DEFAULT_MODEL
            DEFAULT_MODEL = new_model.strip()
            print(f"Model updated to: {DEFAULT_MODEL}")

        # Update global weights
        global DEFAULT_WEIGHTS
        DEFAULT_WEIGHTS = {k: v for k, v in data.items() if k in weight_keys or k == 'starter_code'}

        return jsonify({"message": "Settings updated successfully", "weights": DEFAULT_WEIGHTS, "current_model": DEFAULT_MODEL})

    except Exception as e:
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
