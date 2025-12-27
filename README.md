Project Setup Instructions

Follow the steps below to set up and run the project locally.

1. Clone the Repository
git clone <your-github-repo-link>
cd <project-folder>

2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

After activation, your terminal should show (venv).

3. Install Required Dependencies

Make sure you have pip updated:

pip install --upgrade pip

Install dependencies from requirements.txt:

pip install -r requirements.txt

4. Unzip the Model File

The trained model is provided as a ZIP file.

Navigate to the models/ directory

Unzip the model file:

Windows (PowerShell)
Expand-Archive mudra_classifier.zip


unzip mudra_classifier.zip

5. Run the Flask Application

From the project root directory:

python app.py

(or python main.py if that’s your entry file)

Notes

Always activate the virtual environment before running the project

If dependencies fail to install, ensure your Python version is 3.9 or 3.10

The venv/ folder and model files are excluded from GitHub by design
