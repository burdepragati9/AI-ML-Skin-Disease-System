# Skin Disease Detection System

A deep learning based Streamlit web app that detects skin diseases from uploaded images using TensorFlow.

## Features

- Upload a skin image in JPG, JPEG, or PNG format
- Predict the disease using a trained CNN model
- Show confidence score
- Display all class probabilities
- Use Gemini Vision as an optional fallback when CNN confidence is low
- Simple Streamlit user interface

## Project Structure

```text
New_Skin_Diseases_Project/
+-- app.py
+-- clean_dataset.py
+-- crop_dataset.py
+-- requirements.txt
+-- README.md
+-- model/
    +-- skin_model.keras
    +-- class_names.json
    +-- train.py
    +-- predict.py
```

## How to Run in VS Code

### 1. Open the Project

Open VS Code, then open this folder:

```text
D:\New_Skin_Diseases_Project\New_Skin_Diseases_Project
```

Make sure `app.py` and `requirements.txt` are visible in the VS Code Explorer.

### 2. Open the Terminal

In VS Code, open:

```text
Terminal > New Terminal
```

The terminal should be opened inside the project folder.

### 3. Create a Virtual Environment

Run this command:

```powershell
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the virtual environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Required Packages

Run:

```powershell
pip install -r requirements.txt
```

### 6. Add Gemini API Key

The low-confidence AI fallback uses Gemini Vision. Create this file:

```text
.streamlit/secrets.toml
```

Add your Google Gemini API key from Google AI Studio:

```toml
GOOGLE_API_KEY = "your-gemini-api-key"
```

You can also set `GOOGLE_API_KEY` or `GEMINI_API_KEY` as an environment variable.

### 7. Check Model Files

Before running the app, make sure these files exist:

```text
model/skin_model.keras
model/class_names.json
```

If these files are missing, train the model first:

```powershell
python .\model\train.py
```

### 8. Run the Streamlit App

Run:

```powershell
streamlit run app.py
```

After running the command, Streamlit will show a local URL like:

```text
http://localhost:8501
```

Open that URL in your browser to use the app.

## Notes

- Use clear skin images for better prediction results.
- This project is only an AI demo and is not a medical diagnosis.
- Always consult a doctor for real medical advice.
- Run `python clean_dataset.py --help`, `python crop_dataset.py --help`, or `python model\predict.py --help` to see utility script options.
