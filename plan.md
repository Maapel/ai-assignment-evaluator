Project Plan: AI Assignment Evaluator (Local-Only)

1. Project Overview

1.1. Goal

Create a simple, local-only web application to evaluate student .zip files. The instructor runs a local Python server, uses a web page to upload files, and sees the results on that web page.

1.2. Core Features

Local Server: A single Python Flask file (app.py) that runs on your machine.

Frontend UI: A single HTML file (ai_evaluator.html) that provides the web interface.

File Upload: The web page will have a component for uploading .zip files.

Background Evaluation: The Flask server uses a new thread for each job, so the browser doesn't time out.

Functional Grading: Runs student code against test cases in a secure sandbox (Docker).

Similarity Detection: Compares submissions for plagiarism.

AI Analysis: Uses the Gemini API for code quality review.

Report Dashboard: The web page polls the local server for the report.

2. System Architecture (Local-Only)

2.1. Architectural Diagram

Frontend (Browser) -> Uploads Files -> Local Flask Server (localhost:5000)
Local Flask Server -> Returns {job_id} -> Frontend (Browser)
Local Flask Server -> (Starts new Thread)
Frontend (Browser) -> Polls /status/{job_id} -> Local Flask Server

Inside the Server's Background Thread:

Thread -> Runs Analysis (Docker, Gemini)

Thread -> Saves Report -> Global Dictionary (in-memory)

When Frontend polls:
Local Flask Server -> Reads Report -> Global Dictionary
Local Flask Server -> Returns Report -> Frontend (Browser)

2.2. Component Descriptions

Frontend (Client):

Description: A static HTML file (ai_evaluator.html) that the agent will create. It provides the UI for file upload and polling the backend for reports.

Technology: HTML, Tailwind CSS, JavaScript (Fetch API).

Backend Server:

Description: A single Python file (app.py) using the Flask web framework. It handles file uploads, runs analysis in background threads, and stores results in a simple Python dictionary.

Technology: Python (Flask).

Database (In-Memory):

Description: A single global Python dictionary variable inside app.py (e.g., JOBS_DB = {}). This stores the status and report for each job.

Note: All reports will be lost when the server is stopped. This is perfect for simple, local use.

Services & Environments:

Docker Desktop: Must be running on your local machine.

Google Gemini API: External API for AI analysis.

3. Technology Stack (Local-Only)

Component

Technology

Key Libraries/Tools

Frontend

HTML/CSS/JS

HTML, Tailwind CSS, Lucide (Agent to create ai_evaluator.html)

Backend

Python

Flask, python-multipart, docker, pytest, ast, difflib, google-generativeai, threading

Database

Python

A single global dict variable

4. API Endpoint Definitions (Local Flask Server)

Base URL: http://127.0.0.1:5000

1. POST /evaluate

Description: Submits .zip files for evaluation. Starts a background thread.

Request Body: multipart/form-data with one or more files.

Response (202 Accepted):

{ "job_id": "a-unique-job-id-12345" }


2. GET /evaluate/status/{job_id}

Description: Polls for the status and result of an evaluation job.

URL Parameter: job_id.

Response (200 OK): (Same as before)

Processing: { "status": "processing", ... }

Complete: { "status": "complete", "report": { ... } }

Failed: { "status": "failed", ... }

5. Data Models (JSON Schemas)

The JSON structure for the Report and StudentReport are exactly the same as in the previous plan. The only difference is that the main Job object is just a Python dictionary, not a database document.

Job (in JOBS_DB dictionary):

JOBS_DB = {
    "job-id-123": {
        "status": "pending | processing | complete | failed",
        "status_message": "Starting evaluation...",
        "report": null | { ... } # The final report
    }
}


6. Backend Implementation Plan (Flask)

This will all be in one file, app.py.

Setup (Global):

Import Flask, threading, and all analysis libraries.

Initialize Flask app: app = Flask(__name__).

Enable CORS (so the HTML file can talk to the server): CORS(app).

Create the in-memory database: JOBS_DB = {}.

Define the Main Evaluation Function:

Create a normal Python function (not async): def run_evaluation(job_id: str, file_paths: list)

This function does all the work:

Setup: JOBS_DB[job_id]["status"] = "processing".

Unzip: Unzip files to a temp directory.

Grade: Loop and run code in Docker.

Plagiarism Check: Use ast and difflib.

AI Analysis: Loop and call Gemini API.

Report: Compile the final JSON report.

Save: JOBS_DB[job_id]["status"] = "complete" and JOBS_DB[job_id]["report"] = final_report.

Wrap this entire function in a try...except block. On failure, update JOBS_DB with status: "failed".

Create the /evaluate Endpoint:

Define @app.route("/evaluate", methods=["POST"]).

Inside the function:

Generate a unique job_id (e.g., f"job-{uuid.uuid4()}").

Save the uploaded files to a temporary location.

Create the initial job: JOBS_DB[job_id] = {"status": "pending", "status_message": "Queued...", "report": null}.

Start the background thread:

thread = threading.Thread(target=run_evaluation, args=(job_id, temp_file_paths))
thread.start()


Immediately return jsonify({"job_id": job_id}), 202.

Create the /evaluate/status Endpoint:

Define @app.route("/evaluate/status/<job_id>", methods=["GET"]).

Inside the function:

Find the job: job = JOBS_DB.get(job_id).

If not found, return a 404 error.

Return jsonify(job).

Run the Server:

Add the standard Flask run block at the end of app.py:

if __name__ == "__main__":
    app.run(debug=True, port=5000)


7. Frontend Implementation Plan (ai_evaluator.html)

This file needs to be created by the agent. It will be a single HTML file containing all necessary HTML, CSS (via Tailwind CDN), and JavaScript.

7.1. HTML Structure

A main container (<div class="container mx-auto p-8">).

A header section with the title: <h1>AI Assignment Evaluator</h1>.

A "File Upload" section (<div id="upload-container">) with a drag-and-drop zone and a file input.

An "Upload" button (<button id="upload-btn">).

A "Loading" section (<div id="loading-spinner">), initially hidden, to show a spinner and status text.

A "Results" section (<div id="results-container">), initially hidden, to display summary statistics and individual student reports.

7.2. CSS (Tailwind)

Load Tailwind via the CDN in the <head>: <script src="https://cdn.tailwindcss.com"></script>.

Load Lucide icons in the <head>: <script src="https://unpkg.com/lucide@latest"></script>.

Use Tailwind classes for a modern, responsive layout (flexbox, grid, rounded corners, shadows).

Style the file upload zone (border-dashed), buttons (bg-blue-500, rounded-lg), and report cards (bg-white, shadow-md, rounded-lg).

Add a hidden class to the loading and results sections initially.

7.3. JavaScript Logic

File Handling: Get the list of File objects from the drag-and-drop events or the file input.

Event Listener: Add a click listener to the #upload-btn.

uploadFiles() Function (called on button click):

Creates a FormData object.

Appends all selected files to it.

Disables the upload button and shows the #loading-spinner.

Makes a fetch call (POST) to http://127.0.0.1:5000/evaluate with the FormData.

On success, gets the job_id from the response.

Calls pollForResults(job_id).

Handles fetch errors.

pollForResults(job_id) Function:

Uses setInterval to call fetch (GET) on http://127.0.0.1:5000/evaluate/status/{job_id} every 2-3 seconds.

Inside the interval:

If response.status === "processing", update the status message (e.g., loading-spinner.textContent = response.status_message).

If response.status === "complete", stop the interval (clearInterval), hide the #loading-spinner, and call renderReport(response.report).

If response.status === "failed", stop the interval, hide the spinner, and show an error message.

renderReport(report) Function:

Takes the final JSON report object.

Selects the #results-container.

Dynamically creates HTML elements to display the report.summary.

Loops through each student_report in report.student_reports to create a new div (a styled card) for each student, populating it with their data.

Makes the #results-container visible.

Icon Activation: Call lucide.createIcons() after the page loads and after new icons are added dynamically.