import { useEffect, useState, useRef } from 'react';
import Layout from '../../components/layout/Layout';
import api from '../../services/api';
import './ImagePrediction.css';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];
const MAX_IMAGES = 4;
const MIN_IMAGES = 1;

export default function ImagePrediction() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [predictClicked, setPredictClicked] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [dashboardLoading, setDashboardLoading] = useState(true);
  const [freeSearchesLeft, setFreeSearchesLeft] = useState(4);
  const [results, setResults] = useState([]);
  const [expandedCard, setExpandedCard] = useState(null);
  const [showFaceDetectionDialog, setShowFaceDetectionDialog] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);

  // Refs to prevent duplicate API calls
  const faceDetectionInProgressRef = useRef(false);
  const predictionInProgressRef = useRef(false);
  const processedFileIdsRef = useRef(new Set());

  const selectedCount = selectedFiles.length;

  const canPredict =
    selectedCount >= 1 &&
    selectedCount <= MAX_IMAGES &&
    freeSearchesLeft > 0 &&
    !predictClicked &&
    !loading &&
    !dashboardLoading;

  // -------------------------
  // CLEAR ALL
  // -------------------------
  const clearAll = () => {
    setPredictClicked(false);
    setResults([]);
    setErrorMsg('');
    setExpandedCard(null);
    setFaceDetected(false);
    setShowFaceDetectionDialog(false);
    
    // Reset refs
    faceDetectionInProgressRef.current = false;
    predictionInProgressRef.current = false;
    processedFileIdsRef.current.clear();

    setSelectedFiles((prev) => {
      prev.forEach((f) => {
        if (f.previewUrl) URL.revokeObjectURL(f.previewUrl);
      });
      return [];
    });
  };

  // -------------------------
  // LOAD DASHBOARD
  // -------------------------
  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setDashboardLoading(true);
      try {
        const res = await api.get('/dashboard/doctor');
        const data = res?.data || {};
        const left = Number(data.free_searches_left ?? 0);

        if (!cancelled) setFreeSearchesLeft(Number.isFinite(left) ? left : 0);
      } catch (e) {
        console.warn('[Dashboard Load Error]', e?.message);
      } finally {
        if (!cancelled) setDashboardLoading(false);
      }
    }

    loadDashboard();
    return () => (cancelled = true);
  }, []);


  // -------------------------
  // FILE HANDLING
  // -------------------------
  const acceptFiles = (filesList) => {
    const files = Array.from(filesList || []);
    if (!files.length) return;

    console.log('[File Upload] Image selected');
    console.log('[File Upload] Files selected:', files.length);
    files.forEach(f => console.log('[File Upload] File name:', f.name, 'Type:', f.type, 'Size:', f.size));

    const filtered = files.filter((f) => ACCEPTED_TYPES.includes(f.type));
    if (filtered.length !== files.length) {
      clearAll();
      return;
    }

    const remaining = MAX_IMAGES - selectedFiles.length;
    const toAdd = filtered.slice(0, remaining);
    const mapped = toAdd.map((f) => ({
      id: `${f.name}-${f.size}-${f.lastModified}`,
      file: f,
      name: f.name,
      size: f.size,
      previewUrl: URL.createObjectURL(f),
    }));
    const nextFiles = [...selectedFiles, ...mapped];

    mapped.forEach((f) => console.log('[File Upload] Created blob URL for', f.name, ':', f.previewUrl));
    setSelectedFiles(nextFiles);

    setPredictClicked(false);
    setResults([]);
    setErrorMsg('');
    setFaceDetected(false);
    setShowFaceDetectionDialog(false);
  };

  const removeOne = (id) => {
    const target = selectedFiles.find((x) => x.id === id);
    if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);

    const nextFiles = selectedFiles.filter((x) => x.id !== id);
    setSelectedFiles(nextFiles);
    setFaceDetected(false);
    setShowFaceDetectionDialog(false);
  };

  // -------------------------
  // 🔥 FIXED API CALL (IMPORTANT)
  // -------------------------
  const predictSingleImage = async (fileObj, abortSignal, consent) => {
  const formData = new FormData();
  formData.append("file", fileObj.file);
  formData.append("consent_for_training", consent.toString());

  const res = await api.post(
  "/predict/predict",
  formData,
  {
    signal: abortSignal
  }
);

  console.log("[PREDICT RESPONSE]", res.data);

  const json = res?.data;

  if (!json || json.status !== "success") {
    throw new Error(
      json?.detail ||
      json?.message ||
      "Backend error"
    );
  }

  return json;
};

  // -------------------------
  // PREDICT HANDLER
  // -------------------------
  const handlePredict = async () => {
    // Guard against duplicate calls
    if (faceDetectionInProgressRef.current) {
      console.log('[Face Detection] Detection already in progress, ignoring duplicate call');
      return;
    }
    
    if (selectedFiles.length === 0) {
      console.log('[Face Detection] No files selected, cannot proceed');
      return;
    }

    const fileId = selectedFiles[0].id;
    console.log('[Face Detection] Calling backend to detect face...');
    console.log('[Face Detection] File ID:', fileId);
    console.log('[Face Detection] Filename:', selectedFiles[0].name);
    
    faceDetectionInProgressRef.current = true;
    
    try {
      // Check first image for face detection
      const formData = new FormData();
      formData.append('file', selectedFiles[0].file);
      
      const res = await api.post('/predict/detect-face', formData);
      
      console.log('[Face Detection] Full API response:', res.data);
      
      // Only proceed if response is successful (HTTP 200)
      if (res.status === 200 && res.data) {
        const faceDetectedResult = res.data.face_detected || false;
        
        console.log('[Face Detection] Backend response - face_detected:', faceDetectedResult);
        setFaceDetected(faceDetectedResult);
        
        if (faceDetectedResult) {
          console.log('[Face Detection] Face detected, showing consent dialog');
          setShowFaceDetectionDialog(true);
        } else {
          console.log('[Face Detection] No face detected, proceeding directly to prediction');
          await performPrediction(false);
        }
      } else {
        throw new Error('Unexpected response from face detection service');
      }
    } catch (error) {
      console.error('[Face Detection] Face detection failed:', error);
      console.error('[Face Detection] Error response:', error.response?.data);
      console.error('[Face Detection] Error status:', error.response?.status);
      
      // On HTTP 500 or any error, stop prediction and show error message
      let errorMsg;
      if (error.response?.status === 500) {
        errorMsg = 'Face detection service is temporarily unavailable. Please try again.';
      } else if (error.response?.status === 401 || error.response?.status === 403) {
        errorMsg = 'Authentication error. Please log in again.';
      } else {
        errorMsg = error.response?.data?.detail || error.message || 'Unable to verify whether the image contains a face. Please try again.';
      }
      
      setErrorMsg(errorMsg);
      // Do NOT call performPrediction on error
    } finally {
      faceDetectionInProgressRef.current = false;
    }
  };

  const handleFaceDetectionProceed = () => {
    console.log('[Face Detection] User clicked Continue');
    setShowFaceDetectionDialog(false);
    // Continue prediction and save image for future retraining
    performPrediction(true);
  };

  const handleFaceDetectionCancel = () => {
    console.log('[Face Detection] User clicked Cancel');
    setShowFaceDetectionDialog(false);
    // Continue prediction but do NOT save image for retraining
    performPrediction(false);
  };

  const performPrediction = async (consent) => {
    // Guard against duplicate prediction calls
    if (predictionInProgressRef.current) {
      console.log('[Prediction] Prediction already in progress, ignoring duplicate call');
      return;
    }
    
    console.log('[Prediction] performPrediction executed');
    console.log('[Prediction] performPrediction consent value:', consent);
    
    predictionInProgressRef.current = true;
    setPredictClicked(true);
    setLoading(true);
    setErrorMsg('');
    setResults([]);

    const controller = new AbortController();

    try {
      const perImageResults = [];

      for (const f of selectedFiles) {
        // Skip if this file was already processed
        if (processedFileIdsRef.current.has(f.id)) {
          console.log('[Prediction] File already processed, skipping:', f.id);
          continue;
        }
        
        console.log('[Prediction] Processing file:', f.name, 'with consent:', consent);
        const prediction = await predictSingleImage(f, controller.signal, consent);

        // Mark file as processed
        processedFileIdsRef.current.add(f.id);

        perImageResults.push({
          id: f.id,
          imagePreviewUrl: f.previewUrl,
          imageName: f.name,
          timestamp: new Date().toISOString(),
          prediction,
        });
      }

      setResults(perImageResults);
    } catch (e) {
      console.error('[Prediction] Prediction failed:', e);
      console.error('[Prediction] Error status:', e.response?.status);
      
      // Handle specific error cases
      if (e.response?.status === 403) {
        const detail = e.response?.data?.detail || '';
        if (detail.includes('limit') || detail.includes('free')) {
          setErrorMsg('You have reached your prediction limit.');
        } else {
          setErrorMsg('Authorization error. Please log in again.');
        }
      } else {
        setErrorMsg(e?.message || 'Prediction failed');
      }
      
      setResults([]);
      setPredictClicked(false);
    } finally {
      setLoading(false);
      predictionInProgressRef.current = false;
    }
  };

  // -------------------------
  // UI
  // -------------------------
  return (
    <Layout
      title="Image Prediction"
      userLabel="Doctor"
      role={localStorage.getItem('role') || 'doctor'}
    >
      <div className="ipPage">
        <div className="ipHero">
          <div>
            <div className="ipTitle">Skin Disease Detection</div>
            <div className="ipSubtitle">
              Upload up to 4 images for prediction. Use drag & drop for faster selection.
            </div>
          </div>
          <div className="ipHeroTag">
            {freeSearchesLeft} free search{freeSearchesLeft === 1 ? '' : 's'} left
          </div>
        </div>

        <div className="ipSection glassCard">
          <div className="ipSectionHeader">
            <div className="ipSectionTitle">📤 Upload Images</div>
            <div className="ipSectionHint">Accepted: .jpg / .jpeg / .png</div>
          </div>

          {/* Drag & Drop */}
          <div
            className="ipDropZone"
            role="button"
            tabIndex={0}
            onClick={() => document.getElementById('ipFileInput')?.click()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                document.getElementById('ipFileInput')?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer?.files) acceptFiles(e.dataTransfer.files);
            }}
          >
            <input
              id="ipFileInput"
              className="ipFileInput"
              type="file"
              multiple
              accept=".jpg,.jpeg,.png"
              onChange={(e) => acceptFiles(e.target.files)}
            />

            <div className="ipDropInner">
              <div className="ipDropIcon">📁</div>

              <div className="ipDropText">
                <div className="ipDropHeading">Drag & drop your images</div>
                <div className="ipDropMeta">or click to browse files</div>
              </div>

              <div className="ipDropCount" aria-label="Selected images count">
                <span className="ipDropCountBig">{selectedCount}</span>
                <span className="ipDropCountSmall"> / {MAX_IMAGES}</span>
              </div>
            </div>
          </div>

          {/* Selected previews */}
          {selectedFiles.length > 0 && (
            <div className="ipPreviewLarge">
              {selectedFiles.map((f) => (
                <div key={f.id} className="ipPreviewItem">
                  <div className="ipPreviewThumb">
                    <img src={f.previewUrl} alt={f.name} />
                  </div>

                  <div className="ipPreviewMeta">
                    <div className="ipPreviewName" title={f.name}>
                      {f.name}
                    </div>
                    <div className="ipPreviewSize">{(f.size / 1024).toFixed(1)} KB</div>
                  </div>

                  <button
                    type="button"
                    className="ipRemoveBtn"
                    onClick={() => removeOne(f.id)}
                    disabled={loading}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="ipActionRow">
            <button
              type="button"
              className="ipBtn ipBtnPrimary"
              onClick={handlePredict}
              disabled={!canPredict}
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>

            <button
              type="button"
              className="ipBtn ipBtnGhost"
              onClick={clearAll}
              disabled={loading}
            >
              Clear
            </button>
          </div>

          {errorMsg && (
            <div style={{ marginTop: 12, color: 'red', fontWeight: 800 }}>{errorMsg}</div>
          )}
        </div>

        {/* Face Privacy Consent Dialog */}
        {showFaceDetectionDialog && (
          <div className="ipConsentOverlay">
            <div className="ipConsentDialog glassCard">
              <div className="ipConsentTitle">Face Privacy Consent</div>
              <div className="ipConsentMessage">
                This uploaded image contains a visible human face.
                <br /><br />
                Do you want to continue with disease prediction?
                <br /><br />
                Continue: Prediction will save image for future AI retraining.<br />
                Cancel: Prediction will NOT save image for retraining.
              </div>
              <div className="ipConsentActions">
                <button
                  type="button"
                  className="ipBtn ipBtnPrimary"
                  onClick={handleFaceDetectionProceed}
                >
                  Continue
                </button>
                <button
                  type="button"
                  className="ipBtn ipBtnGhost"
                  onClick={handleFaceDetectionCancel}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}


        {/* Empty State */}
        {selectedFiles.length === 0 && !loading && results.length === 0 && (
          <div className="ipSection glassCard">
            <div className="ipEmptyState">
              <div className="ipEmptyStateIcon">🔬</div>
              <div className="ipEmptyStateTitle">No Images Uploaded Yet</div>
              <div className="ipEmptyStateText">
                Upload skin condition images to get AI-powered disease predictions. 
                Our medical AI system will analyze your images and provide detailed diagnostic insights.
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="ipSection glassCard">
            <div className="ipSectionHeader">
              <div className="ipSectionTitle">🎯 Prediction Results</div>
              <div className="ipSectionHint">
                {results.length} image{results.length === 1 ? '' : 's'} processed
              </div>
            </div>

            <div className="ipResultGrid">
              {results.map((r) => {
                const predicted = r.prediction?.predicted_disease ?? 'Unknown';
                const confidenceNum = Number(r.prediction?.confidence ?? 0);
                const confidencePct = Number.isFinite(confidenceNum) ? confidenceNum : 0;
                const isExpanded = expandedCard === r.id;
                const modelPredictions = r.prediction?.model_predictions || [];
                const majorityVote = r.prediction?.majority_vote || {};
                const aiVerification = r.prediction?.ai_verification || {};
                const predictionSource = r.prediction?.prediction_source || 'unknown';
                const finalClass = aiVerification?.final_class || predicted;
                const finalConfidence = aiVerification?.final_confidence || confidencePct;

                console.log('[Prediction UI debug] prediction.prediction_source =', r?.prediction?.prediction_source);
                console.log('[Prediction UI debug] timestamp =', r?.timestamp);

                const getSourceBadgeClass = (source) => {
                  const classes = {
                    ai_verification: 'ipSourceBadge--ai-verification',
                    majority_voting: 'ipSourceBadge--majority-voting',
                    unanimous_voting: 'ipSourceBadge--unanimous-voting',
                    ensemble_voting: 'ipSourceBadge--ensemble-voting'
                  };
                  return classes[source] || '';
                };

                const badgeClass = getSourceBadgeClass(predictionSource);

                return (
                  <div key={r.id} className="ipPredCard ipPredCard--result">
                    <div className="ipPredTop">
                      <div className="ipPredThumbLarge">
                        <img src={r.imagePreviewUrl} alt={r.imageName} />
                      </div>

                      <div className="ipPredNameBlock">
                        <div className="ipPredName" title={r.imageName}>
                          {r.imageName}
                        </div>
                        <div className="ipPredSubtle">{new Date(r.timestamp).toLocaleString()}</div>
                      </div>
                    </div>

                    <div className="ipPredRows">
                      <div className="ipInfoRow">
                        <div className="ipInfoLabel">Disease</div>
                        <div className="ipInfoValue">
                          <span className="ipDiseaseBadge">{predicted}</span>
                        </div>
                      </div>

                      <div className="ipInfoRow ipInfoRowTight">
                        <div className="ipInfoLabel">Confidence</div>
                        <div className="ipInfoValue ipConfidence">{confidencePct}%</div>
                      </div>

                      <div className="ipInfoRow">
                        <div className="ipInfoLabel">Prediction Source</div>
                        <div className="ipInfoValue">
                          <span className={`ipSourceBadge ${badgeClass}`}>{predictionSource.replace(/_/g, ' ')}</span>
                        </div>
                      </div>


                      <div className="ipConfidenceBar">
                        <div className="ipConfidenceTrack">
                          <div
                            className="ipConfidenceFill"
                            style={{ width: `${Math.max(0, Math.min(100, confidencePct))}%` }}
                          />
                        </div>
                      </div>

                      <button
                        type="button"
                        className="ipBtn ipBtnGhost"
                        onClick={() => setExpandedCard(isExpanded ? null : r.id)}
                        style={{ marginTop: '12px', fontSize: '13px', padding: '10px 16px' }}
                      >
                        {isExpanded ? '▼ Hide Model Comparison' : '▶ View Model Comparison'}
                      </button>
                    </div>

                    {isExpanded && (
                      <div style={{ marginTop: '14px' }}>
                        {/* Multi-Model Outputs */}
                        {modelPredictions.length > 0 && (
                          <div className="ipMultiModelGrid">
                            {modelPredictions.map((mp, idx) => (
                              <div key={idx} className="ipModelCompareCard">
                                <div className="ipModelCompareHeader">
                                  <div className="ipModelCompareTitle" style={{ color: '#111827' }}>
                                    {mp.model_name || `Model ${idx + 1}`}
                                  </div>
                                </div>
                                <div className="ipModelRows">
                                  <div className="ipModelRow">
                                    <div className="ipModelLabel" style={{ color: '#374151' }}>Prediction</div>
                                    <div className="ipModelValue" style={{ color: '#111827', fontWeight: 600 }}>
                                      {mp.predicted_class || 'Unknown'}
                                    </div>
                                    <div className="ipModelConf" style={{ color: '#4B5563', fontWeight: 600 }}>
                                      {Number(mp.confidence || 0).toFixed(1)}%
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Majority Vote */}
                        {majorityVote && Object.keys(majorityVote).length > 0 && (
                          <div className="ipModelCompareCard" style={{ marginTop: '12px' }}>
                            <div className="ipModelCompareHeader">
                              <div className="ipModelCompareTitle" style={{ color: '#111827' }}>
                                Majority Vote
                              </div>
                            </div>
                            <div className="ipModelCompareFooter">
                              <div className="ipModelCompareLine" style={{ color: '#111827' }}>
                                {majorityVote.selected_class || 'N/A'} ({majorityVote.vote_count || 0} votes)
                              </div>
                            </div>
                          </div>
                        )}

                        {/* AI Verification Summary */}
                        {aiVerification && Object.keys(aiVerification).length > 0 && (
                          <div className="ipVerificationCard" style={{ marginTop: '12px' }}>
                            <div className="ipVerificationTitle" style={{ color: '#111827' }}>
                              AI Verification Summary
                            </div>
                            <div className="ipVerificationText" style={{ color: '#111827' }}>
                              {aiVerification.summary || 'No AI verification summary available.'}
                            </div>
                          </div>
                        )}

                        {/* Final Class and Prediction Source */}
                        <div className="ipFinalEnsemble" style={{ marginTop: '12px' }}>
                          <div className="ipFinalEnsembleDisease">
                            <span className="ipFinalBadge" style={{ color: '#111827' }}>
                              Final Class: {finalClass}
                            </span>
                          </div>
                          <div className="ipFinalEnsembleConf">
                            <div className="ipFinalConfLabel" style={{ color: '#374151' }}>
                              Prediction Source
                            </div>
                            <div className="ipFinalConfValue" style={{ color: '#111827', fontSize: '14px' }}>
                              {predictionSource}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
