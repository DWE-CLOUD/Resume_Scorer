# Resume Scorer (Automated Resume Matching System)

![image](https://github.com/user-attachments/assets/9d5405d1-c993-4139-a20e-6de8ddc2618d)


This project implements an **Automated Resume Matching System** that matches resumes to job descriptions using machine learning models and various NLP techniques. It evaluates resumes based on multiple scoring methods such as **TF-IDF**, **BERT embeddings**, and machine learning classifiers like **SVM** and **Naive Bayes**, while also leveraging an **Advanced Generative Model** for content generation and evaluation.

## Key Features

- **Resume Parsing**: Extracts text from PDF and DOCX files.
- **Text Preprocessing**: Cleans and transforms resume text into a usable format.
- **Keyword Matching**: Extracts and matches job description keywords with the resume using TF-IDF and BERT models.
- **Multiple Scoring Models**:
  - **TF-IDF-based similarity score**
  - **BERT embeddings similarity score**
  - **SVM (Support Vector Machine) score**
  - **Naive Bayes score**
  - **Logistic Regression score**
- **Section Extraction**: Identifies key sections in the resume like Skills, Education, and Experience.
- **Link Validation**: Checks if the URLs mentioned in the resume are valid and retrieves their content.
- **Generative Model Integration**: Uses an **Advanced Generative Model** to generate content and scores for advanced resume-job matching.
- **Layout Analysis**: Evaluates the resume's layout by checking for important sections using Optical Character Recognition (OCR).

## Tools and Libraries Used

### 1. **PDF and DOCX Parsing**
   - **pdfplumber**: Extracts text from PDF resumes.
   - **python-docx**: Extracts text from DOCX resumes.

### 2. **Natural Language Processing (NLP)**
   - **nltk**: Provides stopwords for text preprocessing.
   - **re (Regular Expressions)**: Cleans and preprocesses text data.
   - **TF-IDF**: Implemented using `TfidfVectorizer` to extract key features from resumes.
   - **BERT (SentenceTransformer)**: Uses pre-trained BERT embeddings for advanced similarity matching between resumes and job descriptions.

### 3. **Machine Learning Models**
   - **SVM (Support Vector Classifier)**: Pre-trained SVM model (`SVC_model.pkl`) for evaluating resume match.
   - **Naive Bayes**: Pre-trained Gaussian Naive Bayes model (`GaussianNB_model.pkl`).
   - **Logistic Regression**: On-the-fly model trained using job description keywords.
   
### 4. **Web Scraping and URL Validation**
   - **requests**: Retrieves content from URLs listed in resumes.
   - **BeautifulSoup (bs4)**: Extracts and cleans textual content from websites linked in resumes.
   
### 5. **Optical Character Recognition (OCR)**
   - **pdf2image**: Converts PDF pages to images.
   - **pytesseract**: Extracts text from images (PDF pages) for layout analysis.

### 6. **Generative Model**
   - An **Advanced Generative Model** is used to generate content and evaluate the similarity between resumes and job descriptions, scoring each resume based on its content, layout, and structure.

### 7. **Deployment and FastAPI**
   - **FastAPI**: Provides an API for uploading resumes, matching them with job descriptions, and returning a detailed evaluation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DWE-CLOUD/Resume_Scorer
   cd Resume_Scorer
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the necessary NLTK data:

   ```bash
   python -m nltk.downloader stopwords
   ```

4. Ensure that **Tesseract OCR** is installed for layout analysis:

   - On Ubuntu:
     ```bash
     sudo apt install tesseract-ocr
     ```
   - On macOS (with Homebrew):
     ```bash
     brew install tesseract
     ```

## Usage

### API Endpoints

- **Upload Resume and Match with Job Description**: 
  Upload a PDF or DOCX resume and provide a job description. The API will match the resume to the job description and return various scores:
  
  ```bash
  POST /upload/
  ```

  - **Request Body**: 
    - `file`: The resume file (PDF or DOCX).
    - `jd`: The job description string.
  
  - **Response**: A JSON object containing:
    - **ATS Score (TF-IDF)**: Score based on TF-IDF similarity.
    - **BERT Score**: Score using BERT embeddings.
    - **SVM Score**: Predicted score from the SVM model.
    - **NB Score**: Predicted score from the Naive Bayes model.
    - **LR Score**: Predicted score from the Logistic Regression model.
    - **Layout Score**: Score based on resume layout.
    - **Links**: Validated URLs mentioned in the resume.
    - **Sections**: Extracted sections like Skills, Experience, and Education.

- **Root Endpoint**:
  
  ```bash
  GET /
  ```

  Returns a welcome message.

## Project Structure

- **mdo/**: Contains pre-trained machine learning models.
  - `SVC_model.pkl`: Pre-trained Support Vector Classifier model.
  - `GaussianNB_model.pkl`: Pre-trained Naive Bayes model.
- **Notebooks/**: Contains Notebooks for the training.
  - `Start_main.ipnyb`: Contains all the models and their first training and EDA.
  - `Model_training.ipnyb`: Contains the Models on 1000 vector features.
  - `5_feat_change.ipnyb`: Contains the Models on 5 vector features.
  - `1000_to_5_change.ipnyb`: Contains the Models on 1000 to 5 vector change features aka comparison.
- **Dataset/**: Contains Datasets.
  - `URD.csv`: Dataset of the Resume data.
- **temp/**: Contains a space to store resume
  - `----`:
- **requirements.txt**: Dependencies required to run the project.
- **main.py**: Contains the FastAPI implementation.

## Future Work

- **Advanced Hyperparameter Tuning**: Improve model performance by fine-tuning hyperparameters for SVM, Naive Bayes, and Logistic Regression.
- **Deep Learning Models**: Implement more advanced models like LSTM or Transformers for better resume-job matching.
- **Extended Job Description Parsing**: Parse complex job descriptions and categorize key requirements.
  
