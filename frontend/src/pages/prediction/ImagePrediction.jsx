import { useEffect, useState } from 'react';
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

    const filtered = files.filter((f) => ACCEPTED_TYPES.includes(f.type));
    if (filtered.length !== files.length) {
      clearAll();
      return;
    }

    setSelectedFiles((prev) => {
      const remaining = MAX_IMAGES - prev.length;
      const toAdd = filtered.slice(0, remaining);

      const mapped = toAdd.map((f) => ({
        id: `${f.name}-${f.size}-${f.lastModified}`,
        file: f,
        name: f.name,
        size: f.size,
        previewUrl: URL.createObjectURL(f),
      }));

      return [...prev, ...mapped];
    });

    setPredictClicked(false);
    setResults([]);
    setErrorMsg('');
  };

  const removeOne = (id) => {
    setSelectedFiles((prev) => {
      const target = prev.find((x) => x.id === id);
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((x) => x.id !== id);
    });
  };

  // -------------------------
  // 🔥 FIXED API CALL (IMPORTANT)
  // -------------------------
  const predictSingleImage = async (fileObj, abortSignal) => {
  const formData = new FormData();
  formData.append("file", fileObj.file);

  const token = localStorage.getItem("access_token");

console.log("TOKEN =", localStorage.getItem("access_token"));

console.log("AUTH HEADER =", {
  Authorization: `Bearer ${localStorage.getItem("access_token")}`
});

  const res = await api.post(
  "/predict/predict",
  formData,
  {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access_token")}`
    },
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
    setPredictClicked(true);
    setLoading(true);
    setErrorMsg('');
    setResults([]);

    const controller = new AbortController();

    try {
      const perImageResults = [];

      for (const f of selectedFiles) {
        const prediction = await predictSingleImage(f, controller.signal);

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
      setErrorMsg(e?.message || 'Prediction failed');
      setResults([]);
      setPredictClicked(false);
    } finally {
      setLoading(false);
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