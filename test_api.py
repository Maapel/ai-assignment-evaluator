#!/usr/bin/env python3
"""
Simple test script for the AI Assignment Evaluator API
"""
import requests
import time
import os

BASE_URL = "http://127.0.0.1:5000"

def test_api():
    print("🧪 Testing AI Assignment Evaluator API")
    print("=" * 50)

    # Test 1: Check if server is running
    print("1. Testing server connectivity...")
    try:
        response = requests.get(f"{BASE_URL}/evaluate/status/test")
        if response.status_code == 404:
            print("✅ Server is running (404 for non-existent job is expected)")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running!")
        return

    # Test 2: Upload single file
    print("\n2. Testing file upload...")
    sample_file = "sample_submissions/EE24B032_A3.zip"
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        return

    with open(sample_file, 'rb') as f:
        files = {'files': f}
        response = requests.post(f"{BASE_URL}/evaluate", files=files)

    if response.status_code == 202:
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ File uploaded successfully. Job ID: {job_id}")
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        return

    # Test 3: Poll for results
    print("\n3. Polling for evaluation results...")
    max_attempts = 30  # 30 seconds timeout
    attempt = 0

    while attempt < max_attempts:
        response = requests.get(f"{BASE_URL}/evaluate/status/{job_id}")
        data = response.json()

        status = data.get('status')
        message = data.get('status_message', '')

        print(f"   Status: {status} - {message}")

        if status == 'complete':
            print("✅ Evaluation completed!")
            break
        elif status == 'failed':
            print(f"❌ Evaluation failed: {message}")
            return

        time.sleep(2)
        attempt += 1

    if attempt >= max_attempts:
        print("❌ Evaluation timed out")
        return

    # Test 4: Analyze results
    print("\n4. Analyzing results...")
    report = data.get('report', {})
    summary = report.get('summary', {})
    students = report.get('student_reports', [])

    print(f"📊 Summary:")
    print(f"   - Total submissions: {summary.get('total_submissions', 0)}")
    print(".2f")
    print(".2f")

    print(f"\n👥 Student Reports:")
    for student in students:
        print(f"   Student: {student.get('student_id')} ({student.get('filename')})")
        print(".1f")
        print(".1f")
        print(f"   AI Feedback: {student.get('ai_feedback', 'N/A')[:100]}...")

    print("\n🎉 All tests passed! The API is working correctly.")
    print("\n💡 Next: Open ai_evaluator.html in your browser to test the web interface!")

if __name__ == "__main__":
    test_api()
