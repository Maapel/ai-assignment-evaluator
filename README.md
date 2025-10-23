# 🤖 AI Assignment Evaluator

A comprehensive web application for automated evaluation of programming assignments using advanced AI analysis. Perfect for educators who want to efficiently grade coding assignments with detailed feedback and plagiarism detection.

## ✨ Features

### 🎯 Core Functionality
- **Individual & Batch Processing**: Upload single files or process entire class submissions
- **Moodle Integration**: Supports Moodle assignment export format with student folders
- **Real-time Progress**: Live status updates during evaluation
- **Comprehensive Reports**: Detailed analysis with scores and AI feedback

### 🔧 Technical Features
- **Secure Evaluation**: Runs student code in isolated environments (no Docker required for basic analysis)
- **AI-Powered Analysis**: Uses Groq's advanced language models for code and report quality assessment
- **Plagiarism Detection**: Advanced AST-based similarity analysis with detailed reporting
- **Weighted Scoring**: Customizable scoring weights for different evaluation criteria

### 📊 Scoring System
- **Functional Tests**: Static analysis of code implementation (40% default weight)
- **Code Quality**: AI assessment of structure, best practices, readability (40% default weight)
- **Report Quality**: AI evaluation of documentation and technical writing (20% default weight)
- **Overall Score**: Weighted calculation with letter grade assignment

### 🎨 User Experience
- **Modern UI**: Clean, responsive interface with drag-and-drop uploads
- **Settings Panel**: Customize scoring weights and evaluation parameters
- **Export Options**: Download results as CSV or JSON for gradebook integration
- **Plagiarism Insights**: Click to see which submissions are similar

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key (for AI analysis)
- Modern web browser

### Installation

1. **Clone or download the project**
   ```bash
   git clone https://github.com/yourusername/ai-assignment-evaluator.git
   cd ai-assignment-evaluator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://127.0.0.1:5001/
   ```

## 📖 Usage Guide

### Individual File Evaluation
1. Select "Individual Files" mode
2. Drag & drop student submission zip files
3. Click "Start Evaluation"
4. Review detailed results with AI feedback

### Batch Evaluation
1. Select "Batch Upload" mode
2. Upload a zip file containing multiple student submissions
3. System scans and shows list of found submissions
4. Click "Start Batch Evaluation"
5. Monitor real-time progress for each student
6. View comprehensive class results

### Customizing Scoring
1. Click "Scoring Settings" button
2. Adjust weights for different evaluation criteria
3. Save settings (persisted across sessions)
4. All future evaluations use your custom weights

## 📁 Submission Format

### Individual Files
- **Format**: `{STUDENT_ID}_{ASSIGNMENT}.zip`
- **Example**: `EE24B032_A3.zip`
- **Contents**:
  - Python code file(s) (`.py`)
  - Optional PDF report

### Batch Files (Moodle Export)
- **Structure**: Zip containing student folders
- **Example**:
  ```
  batch_submissions.zip/
  ├── ADITYA TATIPAKA_4671_assignsubmission_file_/
  │   └── Assignment 3 - Keyboard Optimization.zip
  ├── ee24b003 Akshara G_4683_assignsubmission_file_/
  │   └── ee24b003_A3.zip
  └── ... (other student folders)
  ```

## 🎯 Scoring Methodology

### Component Breakdown

**1. Functional Tests (0-10 scale)**
- Algorithm implementation correctness
- Function completeness
- Code execution capability
- **Weight**: 40% (customizable)

**2. Code Quality (0-10 scale)**
- Code structure and organization
- Best practices adherence
- Variable naming and documentation
- Readability and maintainability
- **Weight**: 40% (customizable)

**3. Report Quality (0-10 scale)**
- Content completeness and accuracy
- Technical depth and analysis
- Clarity and presentation
- **Weight**: 20% (customizable)

### Final Score Calculation
```
Overall Score = (Functional × 0.4) + (Code Quality × 0.4) + (Report Quality × 0.2)
Grade = A (90%+), B (80-89%), C (70-79%), D (60-69%), F (<60%)
```

## 🔍 Plagiarism Detection

### How It Works
- **AST Analysis**: Compares code structure, not just text
- **Similarity Scoring**: Identifies submissions with >30% similarity
- **Top Matches**: Shows 3 most similar submissions per student
- **Detailed View**: Click info icon to see similarity percentages

### Example Output
```
🔍 Plagiarism Analysis for EE24B032

Similar Submissions Found:
• EE24B031_A3.zip - 75% similar
• EE24B033_A3.zip - 45% similar
• EE24B028_A3.zip - 38% similar
```

## 📤 Export Options

### CSV Export
- Student ID, component scores, overall score, grade
- Compatible with Excel and Google Sheets
- Perfect for gradebook integration

### JSON Export
- Complete evaluation data
- All AI feedback and analysis
- Raw scores and detailed breakdowns

## 🛠️ API Reference

### Endpoints

**File Evaluation**
- `POST /evaluate` - Submit individual files
- `POST /evaluate/batch` - Submit batch file for scanning
- `POST /evaluate/batch/start/{job_id}` - Start batch processing

**Status & Results**
- `GET /evaluate/status/{job_id}` - Check evaluation progress
- `GET /evaluate/results/{job_id}` - Get complete results
- `GET /evaluate/download/{job_id}` - Download results as JSON

**Settings**
- `GET /settings` - Get current scoring weights
- `POST /settings` - Update scoring configuration

### Response Format
```json
{
  "summary": {
    "total_submissions": 25,
    "average_functional_score": 7.8,
    "average_code_quality_score": 6.9,
    "average_report_score": 8.2,
    "average_plagiarism_score": 0.15
  },
  "student_reports": [
    {
      "student_id": "EE24B032",
      "functional_score": 0.85,
      "code_quality_score": 0.72,
      "report_score": 0.88,
      "plagiarism_score": 0.05,
      "plagiarism_details": [],
      "ai_feedback": "Detailed AI analysis...",
      "filename": "EE24B032_A3.zip"
    }
  ]
}
```

## 🔐 Security & Privacy

- **Local Processing**: All evaluation happens on your machine
- **No Data Upload**: Student code never leaves your computer
- **Isolated Execution**: Code runs in controlled environments
- **API Security**: Groq API calls are encrypted and secure

## 🐛 Troubleshooting

### Common Issues

**"Connection refused" errors**
- Ensure Flask server is running on port 5001
- Check that no other service is using the port

**AI analysis shows "disabled"**
- Verify GROQ_API_KEY is set in `.env` file
- Restart Flask server after adding API key

**Batch processing fails**
- Ensure batch zip contains student folders
- Check that student folders contain zip files
- Verify zip files are not corrupted

**Scoring weights not saving**
- Check browser console for JavaScript errors
- Ensure Flask server has write permissions

### Performance Tips

- **Large batches**: Process in smaller chunks for better performance
- **Memory usage**: Close browser tabs when processing large batches
- **API limits**: Groq has rate limits; space out large evaluations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast and accurate AI analysis
- **Tailwind CSS** for the beautiful UI components
- **Lucide** for the icon set
- **Flask** for the robust web framework

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Happy Grading! 🎓**

The AI Assignment Evaluator makes grading programming assignments faster, fairer, and more insightful. Focus on teaching while the AI handles the heavy lifting of code analysis and feedback generation.
