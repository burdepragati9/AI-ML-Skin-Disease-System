# Skin Disease Detection System

A comprehensive AI/ML-powered Streamlit web application for skin disease detection with self-learning capabilities, doctor authentication, and advanced analytics.

## 🚀 Features

### Core Functionality
- **Skin Disease Prediction**: Upload skin images (JPG, JPEG, PNG) for instant disease classification using a trained CNN model
- **Confidence Scoring**: Displays prediction confidence with visual progress bars
- **Heatmap Visualization**: Grad-CAM heatmaps showing model focus areas
- **Comprehensive Results**: Shows probabilities for all disease classes
- **PDF Reports**: Download detailed prediction reports

### 🤖 Self-Learning ML System
- **Automatic Learning**: When ML confidence is low (<70%), AI fallback (Gemini Vision) provides the final diagnosis
- **Incremental Training**: New AI-recognized images automatically added to dataset and model retrained incrementally
- **Duplicate Detection**: Prevents storing duplicate images using perceptual hashing
- **Background Processing**: Asynchronous training queue prevents blocking main application
- **Training Logs**: Complete audit trail of training sessions with accuracy metrics

### 👨‍⚕️ Doctor Authentication System
- **Secure Registration**: Doctor signup with profile information (name, ID, specialization, clinic, etc.)
- **JWT Authentication**: Secure login/logout with session management
- **Password Recovery**: Forgot password with secure token-based reset
- **Free Search Limit**: Each doctor gets 3 free searches, then upgrade required
- **Usage Tracking**: Real-time monitoring of search usage and limits

### 📊 Doctor Dashboard
- **Search Analytics**: Total searches, remaining free searches, usage statistics
- **Disease Trends**: Most searched diseases with interactive charts
- **Prediction Sources**: Breakdown of ML vs AI predictions
- **Search History**: Paginated history with filters (disease, date, confidence, source)
- **Top Images**: Most frequently searched disease images
- **Profile Management**: Update personal information and upload profile photos

### 📈 Admin Analytics
- **Training Metrics**: AI images added, retrained images, accuracy improvements
- **System Status**: Real-time training queue status (queued, processing, completed, failed)
- **Disease Distribution**: Charts showing disease frequency across all searches
- **Prediction Analytics**: ML vs AI prediction usage statistics
- **AI Images Gallery**: Browse all AI-recognized images with metadata

### 🔒 Security & Performance
- **Image Validation**: Secure file upload with type, size, and content validation
- **Rate Limiting**: API rate limiting to prevent abuse
- **Input Sanitization**: All user inputs sanitized and validated
- **Optimized Storage**: Automatic image compression and optimization
- **Async Processing**: Background jobs for training and heavy operations

### 🛠️ Backend APIs
- `GET /api/training/status` - Current training queue status
- `GET /api/training/newly-learned-count` - Count of newly learned images
- `GET /api/ai-recognized-images` - List of AI-recognized images (admin only)

## 🏗️ Project Structure

```
Updated_Project/
├── app.py                          # Main Streamlit application
├── api.py                          # REST API endpoints
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables
├── README.md                      # This file
├── database/
│   ├── db.py                      # Database connection and utilities
│   └── migrations/
│       └── 001_init.sql           # Database schema
├── model/
│   ├── skin_model.keras           # Trained ML model
│   ├── class_names.json           # Disease class names
│   ├── train.py                   # Model training script
│   └── predict.py                 # Prediction utilities
├── training/
│   └── self_learning.py           # Self-learning system
├── auth/
│   └── doctor_auth.py             # Doctor authentication
├── analytics/
│   └── admin_analytics.py         # Analytics functions
├── history/
│   ├── search_history.py          # Search tracking
│   └── uploads/                   # Uploaded images storage
├── utils/
│   ├── config.py                  # Configuration settings
│   ├── security.py                # Security utilities
│   └── ai_recognition.py          # AI fallback system
├── dataset/                       # Self-learning dataset
├── MySkinData/                    # Main training dataset
└── CleanData/                     # Processed dataset
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd Updated_Project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration

Create `.env` file in project root:
```env
# Google Gemini API Key (for AI fallback)
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=models/gemini-2.5-flash

# Optional: Admin API Token for backend APIs
ADMIN_API_TOKEN=your_secure_token_here

# Optional: Custom paths
SKIN_MODEL_PATH=model/skin_model.keras
CLASS_NAMES_PATH=model/class_names.json
SELF_LEARNING_DATASET_PATH=dataset
APP_DB_PATH=database/app.db
UPLOAD_HISTORY_PATH=history/uploads
PROFILE_PHOTO_PATH=history/profile_photos

# Optional: Tuning parameters
LOW_CONFIDENCE_THRESHOLD=70.0  # App enforces at least 70.0
FREE_SEARCH_LIMIT=3
IMAGE_DUPLICATE_HASH_DISTANCE=4
INCREMENTAL_TRAINING_EPOCHS=2
TRAINING_MAX_ATTEMPTS=3
TRAINING_STALE_MINUTES=30
API_RATE_LIMIT_WINDOW_SECONDS=60
API_RATE_LIMIT_REQUESTS=60
```

### 6. Initialize Database
```bash
python -c "from database.db import init_db; init_db()"
```

### 7. Run the Application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### 8. (Optional) Run API Server
```bash
python api.py
```

API will be available at `http://localhost:8000`

## 🔧 Configuration

### Model Configuration
- **Image Size**: 160x160 pixels
- **Base Model**: MobileNetV2 (alpha=0.35)
- **Training Epochs**: 5 (initial), 2 (incremental)
- **Batch Size**: 2
- **Low Confidence Threshold**: 70%

### Security Settings
- **Max Image Size**: 8MB
- **Allowed Formats**: JPG, JPEG, PNG
- **Rate Limit**: 60 requests per minute
- **Password Requirements**: Minimum 8 characters

### Database Tables
- `doctors` - Doctor profiles and authentication
- `free_search_usage` - Search limits tracking
- `searches` - Search history and analytics
- `ai_predictions` - AI fallback predictions
- `image_hashes` - Duplicate detection
- `training_logs` - Training audit trail
- `training_queue` - Background training jobs

## 🚀 Usage

### For Doctors
1. **Register**: Create account with professional details
2. **Login**: Access secure dashboard
3. **Upload Images**: Upload skin images for diagnosis
4. **View Results**: See predictions with confidence scores
5. **Track Usage**: Monitor free search limits
6. **View History**: Browse past searches and analytics

### For Administrators
1. **Access Analytics**: View system-wide statistics
2. **Monitor Training**: Check self-learning progress
3. **Review AI Images**: Browse AI-recognized cases
4. **System Health**: Monitor training queue status

## 🔄 Self-Learning Workflow

1. **User Upload**: Doctor uploads skin image
2. **ML Prediction**: System predicts using trained model
3. **Confidence Check**: If confidence < 70%, trigger AI fallback
4. **AI Diagnosis**: Gemini Vision provides alternative diagnosis
5. **Image Storage**: New AI-recognized images saved to dataset
6. **Duplicate Check**: Hash-based duplicate prevention
7. **Queue Training**: Image added to training queue
8. **Incremental Training**: Background process retrains model
9. **Model Update**: Updated model deployed automatically
10. **Logging**: All steps logged for audit trail

## 📊 Analytics & Monitoring

### Doctor Dashboard Metrics
- Total searches performed
- Remaining free searches
- Most searched diseases
- ML vs AI prediction distribution
- Search history with pagination
- Top searched images

### Admin Dashboard Metrics
- Total AI-recognized images
- Successfully retrained images
- Most common added disease
- Model accuracy improvements
- Training queue status
- Disease frequency distribution

## 🔐 Security Features

- **Password Hashing**: Werkzeug security for passwords
- **Input Validation**: Comprehensive sanitization
- **File Upload Security**: Type, size, and content validation
- **Rate Limiting**: Prevents API abuse
- **Session Management**: Secure doctor sessions
- **Image Hashing**: Duplicate detection and integrity

## 🎯 Performance Optimizations

- **Incremental Training**: Only retrain on new data
- **Background Processing**: Non-blocking training jobs
- **Image Optimization**: Automatic compression and resizing
- **Database Indexing**: Optimized queries for analytics
- **Lazy Loading**: Efficient data loading for dashboards

## 🐛 Troubleshooting

### Common Issues

**TensorFlow Import Error**
```bash
pip install tensorflow==2.16.1
```

**Database Connection Error**
```bash
python -c "from database.db import init_db; init_db()"
```

**Model Loading Error**
- Ensure `model/skin_model.keras` exists
- Check TensorFlow version compatibility

**AI Fallback Not Working**
- Verify `GOOGLE_API_KEY` in `.env`
- Check API quota and billing

### Logs
- Training logs: Check `training_logs` table
- Application logs: Check terminal output
- Error logs: Check database for failed training jobs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is for educational purposes only. Not intended for medical diagnosis.

## ⚠️ Disclaimer

This application is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

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
