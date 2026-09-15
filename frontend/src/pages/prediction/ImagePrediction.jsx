import { useEffect, useState, useRef } from 'react';
import Layout from '../../components/layout/Layout';
import api from '../../services/api';
import './ImagePrediction.css';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];
const MAX_IMAGES = 4;
const MIN_IMAGES = 1;

let dashboardRequestPromise = null;
let dashboardCache = null;

const createUploadId = () => {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

const loadDoctorDashboardOnce = async () => {
  if (dashboardCache) return dashboardCache;
  if (!dashboardRequestPromise) {
    dashboardRequestPromise = api.get('/dashboard/doctor').then((res) => {
      dashboardCache = res;
      return res;
    }).finally(() => {
      dashboardRequestPromise = null;
    });
  }
  return dashboardRequestPromise;
};

/**
 * Invalidate the cached dashboard response so the next load fetches fresh data
 * (e.g. after predictions consume free searches).
 */
const invalidateDashboardCache = () => {
  dashboardCache = null;
};

export default function ImagePrediction() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [predictClicked, setPredictClicked] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [dashboardLoading, setDashboardLoading] = useState(true);
  const [freeSearchesLeft, setFreeSearchesLeft] = useState(4);
  const [results, setResults] = useState([]);
  const [expandedCard, setExpandedCard] = useState(null);

  // Per-image face detection and consent tracking with promise-based dialog
  const [showFaceDetectionDialog, setShowFaceDetectionDialog] = useState(false);
  const consentResolveRef = useRef(null);
  const [dialogFileName, setDialogFileName] = useState('');

  // Ref to prevent duplicate predict button clicks while a run is in progress.
  // We intentionally do NOT track "processed" file ids across runs, because
  // each Predict click must re-run the full Upload → Face Detection → Consent
  // → Prediction flow for every selected image. Skipping previously processed
  // ids caused missing result cards on retry.
  const predictRunInProgressRef = useRef(false);

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
    setShowFaceDetectionDialog(false);
    consentResolveRef.current = null;

    // Reset the run-in-progress guard so a fresh Predict run can start.
    predictRunInProgressRef.current = false;

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
        const res = await loadDoctorDashboardOnce();
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

    const filtered = files.filter((f) => ACCEPTED_TYPES.includes(f.type));
    if (filtered.length !== files.length) {
      clearAll();
      return;
    }

    const remaining = MAX_IMAGES - selectedFiles.length;
    const toAdd = filtered.slice(0, remaining);
    const mapped = toAdd.map((f) => ({
      id: createUploadId(),
      file: f,
      name: f.name,
      size: f.size,
      previewUrl: URL.createObjectURL(f),
      face_detected: null,
      detector_used: null,
      consent_required: null,
      consent_given: null,
      prediction: null,
    }));
    setSelectedFiles((prev) => [...prev, ...mapped]);

    setPredictClicked(false);
    setResults([]);
    setErrorMsg('');
    setShowFaceDetectionDialog(false);
  };

  const removeOne = (id) => {
    const target = selectedFiles.find((x) => x.id === id);
    if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);

    const nextFiles = selectedFiles.filter((x) => x.id !== id);
    setSelectedFiles(nextFiles);
    setShowFaceDetectionDialog(false);
  };

  // -------------------------
  // 🔥 Challenge user for consent via promise-based dialog
  // -------------------------
  const updateImageState = (id, patch) => {
    setSelectedFiles((prev) => (
      prev.map((item) => (item.id === id ? { ...item, ...patch } : item))
    ));
  };

  const waitForConsent = (fileName) => {
    return new Promise((resolve) => {
      consentResolveRef.current = resolve;
      setDialogFileName(fileName);
      setShowFaceDetectionDialog(true);
    });
  };

  const handleFaceDetectionProceed = () => {
    setShowFaceDetectionDialog(false);
    // User clicked Cancel: consent for training is denied.
    // Disease prediction must still continue, but the image must not
    // be stored or used for future training.
    if (consentResolveRef.current) {
      consentResolveRef.current(false);
      consentResolveRef.current = null;
    }
  };

  const handleFaceDetectionCancel = () => {
    setShowFaceDetectionDialog(false);
    // User clicked Cancel: reset loading state and do not call the prediction
    // API. The prediction flow will detect the false consent value and stop.
    if (consentResolveRef.current) {
      consentResolveRef.current(false);
      consentResolveRef.current = null;
    }
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
  // PREDICT HANDLER with per-image face detection
  // -------------------------
  const handlePredict = async () => {
    // Guard against duplicate concurrent Predict runs (e.g. double-click).
    if (predictRunInProgressRef.current) {
      return;
    }

    if (selectedFiles.length === 0) {
      return;
    }

    predictRunInProgressRef.current = true;
    setPredictClicked(true);
    // NOTE: Do NOT set loading=true here. Loading (and the "Predicting..."
    // button state) must only be enabled once we are actually about to call
    // the prediction API. While the Face Privacy Consent dialog is open, the
    // UI must remain idle so the user can make a decision first.
    setErrorMsg('');
    setResults([]);

    const controller = new AbortController();

    try {
      const perImageResults = [];

      for (const f of selectedFiles) {
        // Step 1: Detect face for this exact image (one intentional call).
        const detectFormData = new FormData();
        detectFormData.append('file', f.file);

        const detectRes = await api.post('/predict/detect-face', detectFormData);

        if (detectRes.status !== 200 || !detectRes.data) {
          throw new Error("Face detection failed for " + f.name);
        }

        // Read the exact backend response contract fields.
        const faceDetectedResult = detectRes.data.face_detected === true;
        const detectorUsed = detectRes.data.detector_used || "unknown";
        const consentRequired = detectRes.data.consent_required === true;

        updateImageState(f.id, {
          face_detected: faceDetectedResult,
          detector_used: detectorUsed,
          consent_required: consentRequired,
        });
        // Step 2: Determine consent for this exact image.
        let consentForThisImage = false;

        if (consentRequired) {
          // Pause the prediction flow completely while the consent dialog is
          // open. Do NOT show a loading spinner, do NOT change the Predict
          // button to "Predicting...", and do NOT call /predict/predict.
          setLoading(false);
          consentForThisImage = await waitForConsent(f.name);

          if (!consentForThisImage) {
            // User clicked Cancel: close popup, reset loading state, and do
            // NOT call the prediction API for this (or any remaining) image.
            updateImageState(f.id, { consent_given: false });
        
          }
        }
        updateImageState(f.id, { consent_given: consentForThisImage });

        // Step 3: Predict for this exact image with its individual consent
        // value. Only now (after consent is given or not required) do we set
        // loading=true and call the prediction API.
        setLoading(true);
        const prediction = await predictSingleImage(f, controller.signal, consentForThisImage);
        updateImageState(f.id, { prediction });

        // Build the result object keyed by the stable image id so the result
        // card is always bound to the exact same image that was uploaded.
        perImageResults.push({
          id: f.id,
          imagePreviewUrl: f.previewUrl,
          imageName: f.name,
          timestamp: new Date().toISOString(),
          prediction,
          faceDetected: faceDetectedResult,
          detectorUsed,
          consentRequired,
          consentGiven: consentForThisImage,
        });
      }

      setResults(perImageResults);

      // Predictions consume free searches; refresh the cached dashboard count.
      invalidateDashboardCache();
      try {
        const res = await loadDoctorDashboardOnce();
        const left = Number(res?.data?.free_searches_left ?? 0);
        setFreeSearchesLeft(Number.isFinite(left) ? left : 0);
      } catch {
        /* non-fatal: leave previous count as-is */
      }
    } catch (e) {
      console.error('[Prediction] Prediction failed:', e);

      // Make sure the consent dialog is never left open after an error.
      setShowFaceDetectionDialog(false);
      if (consentResolveRef.current) {
        consentResolveRef.current(false);
        consentResolveRef.current = null;
      }

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
      predictRunInProgressRef.current = false;
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
                Image: <strong>{dialogFileName}</strong>
                <br /><br />
                This image contains a visible human face.
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

                      <div className="ipInfoRow">
                        <div className="ipInfoLabel">Face Detected / Consent</div>
                        <div className="ipInfoValue">
                          <span className={`ipSourceBadge ${r.faceDetected ? '' : ''}`}>
                            {r.faceDetected ? 'Face ✓' : 'No Face'} / {r.consentGiven ? 'Consent ✓' : 'No Consent'}
                          </span>
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
                                {majorityVote.selected_class || 'N/A'} ({majorityVote.supporting_models?.length || (majorityVote.vote_counts?.[majorityVote.selected_class] || 0)} votes)
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
