import { useEffect, useMemo, useState } from 'react';
import Layout from '../../components/layout/Layout';
import api from '../../services/api';

function LoadingState() {
  return (
    <div className="phCenter" aria-label="Loading">
      <div className="phSpinner" />
      <div className="phMuted" style={{ marginTop: 12 }}>
        Loading prediction history...
      </div>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="phErrorWrap" aria-label="Error">
      <div className="phErrorTitle">Failed to load prediction history</div>
      <div className="phErrorMsg">{message}</div>
      {onRetry && (
        <button type="button" className="phBtn phBtnPrimary" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

function fmtConfidence(confidence) {
  const num = Number(confidence);
  if (Number.isFinite(num)) return `${num.toFixed(1)}%`;
  return confidence ?? 'N/A';
}

function formatTimestamp(timestamp) {
  if (!timestamp) return 'N/A';
  return new Date(timestamp).toLocaleString();
}

function getSourceBadgeColor(source) {
  const colors = {
    ai_verification: { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgba(34, 197, 94, 0.35)', text: '#166534' },
    majority_voting: { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.35)', text: '#1e40af' },
    unanimous_voting: { bg: 'rgba(168, 85, 247, 0.15)', border: 'rgba(168, 85, 247, 0.35)', text: '#7c3aed' },
    ensemble_voting: { bg: 'rgba(249, 115, 22, 0.15)', border: 'rgba(249, 115, 22, 0.35)', text: '#c2410c' }
  };
  return colors[source] || { bg: 'rgba(148, 163, 184, 0.15)', border: 'rgba(148, 163, 184, 0.35)', text: '#374151' };
}

export default function PredictionHistory() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [rows, setRows] = useState([]);
  const [diseaseFilter, setDiseaseFilter] = useState('All Diseases');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const itemsPerPage = 10;

  useEffect(() => {
    let cancelled = false;

    async function loadHistory() {
      setLoading(true);
      setError('');

      try {
        const response = await api.get('/dashboard/doctor');
        const data = response?.data || {};
        const historyRows = Array.isArray(data.prediction_history) ? data.prediction_history : [];

        if (!cancelled) setRows(historyRows);
      } catch (e) {
        if (cancelled) return;
        const msg = e?.response?.data?.detail || e?.message || 'Unexpected error';
        setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadHistory();

    return () => {
      cancelled = true;
    };
  }, []);

  const uniqueDiseases = useMemo(() => {
    const diseases = new Set(rows.map(row => row.predicted_disease).filter(Boolean));
    return ['All Diseases', ...Array.from(diseases).sort()];
  }, [rows]);

  const filteredRows = useMemo(() => {
    return rows.filter(row => {
      const matchesDisease = diseaseFilter === 'All Diseases' || row.predicted_disease === diseaseFilter;
      const searchLower = searchQuery.toLowerCase();
      const matchesSearch = !searchQuery || 
        (row.image_name && row.image_name.toLowerCase().includes(searchLower)) ||
        (row.predicted_disease && row.predicted_disease.toLowerCase().includes(searchLower));
      return matchesDisease && matchesSearch;
    });
  }, [rows, diseaseFilter, searchQuery]);

  const paginatedRows = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredRows.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredRows, currentPage]);

  const totalPages = Math.ceil(filteredRows.length / itemsPerPage);

  const handleRowClick = (row) => {
    setSelectedRecord(row);
    setShowPreview(true);
  };

  const handleClosePreview = () => {
    setShowPreview(false);
    setSelectedRecord(null);
  };

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
  };

  if (loading) {
    return (
      <Layout title="Prediction History" userLabel="Doctor" role="doctor">
        <LoadingState />
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Prediction History" userLabel="Doctor" role="doctor">
        <ErrorState message={error} onRetry={() => window.location.reload()} />
      </Layout>
    );
  }

  return (
    <Layout title="Prediction History" userLabel="Doctor" role="doctor">
      <div className="phPage">
        <header className="phHeader">
          <div className="phTitleWrap">
            <div className="phTitle">Prediction History</div>
            <div className="phSubtitle">Your uploaded images and predictions</div>
          </div>
        </header>

        <div className="phControls">
          <div className="phControlGroup">
            <label className="phLabel">Disease Filter</label>
            <select 
              className="phSelect" 
              value={diseaseFilter} 
              onChange={(e) => {
                setDiseaseFilter(e.target.value);
                setCurrentPage(1);
              }}
            >
              {uniqueDiseases.map(disease => (
                <option key={disease} value={disease}>{disease}</option>
              ))}
            </select>
          </div>

          <div className="phControlGroup">
            <label className="phLabel">Search</label>
            <input 
              type="text" 
              className="phInput" 
              placeholder="Search by image name or disease..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
            />
          </div>
        </div>

        {!filteredRows.length && (
          <div className="phEmpty">
            {searchQuery || diseaseFilter !== 'All Diseases' 
              ? 'No predictions match your filters.' 
              : 'No prediction history found.'}
          </div>
        )}

        {filteredRows.length > 0 && (
          <>
            <div className="phTableWrap">
              <table className="phTable" aria-label="Prediction history">
                <thead>
                  <tr>
                    <th>Image Name</th>
                    <th>Predicted Disease</th>
                    <th>Confidence Score</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedRows.map((row, idx) => {
                    const badgeColor = getSourceBadgeColor(row.prediction_source);
                    return (
                      <tr 
                        key={`${row.timestamp || idx}-${idx}`} 
                        className="phRow"
                        onClick={() => handleRowClick(row)}
                      >
                        <td className="phPrimaryText">
                          {(() => {
                            const getResolvedImageUrl = (item) => {
                              if (item?.image_url) return item.image_url;

                              const imagePath = item?.image_path;
                              if (imagePath) {
                                if (
                                  typeof imagePath === 'string' &&
                                  (imagePath.startsWith('http://') ||
                                    imagePath.startsWith('https://'))
                                ) {
                                  return imagePath;
                                }

                                // Backend might provide a filesystem path.
                                // Reuse the same upload mapping strategy used elsewhere.
                                const marker = 'history/uploads/';
                                const idx = String(imagePath).replace(/\\/g, '/').indexOf(marker);
                                if (idx !== -1) {
                                  const relPart = String(imagePath)
                                    .replace(/\\/g, '/')
                                    .substring(idx + marker.length);
                                  return `http://127.0.0.1:8000/uploads/${relPart}`;
                                }
                              }

                              return item?.image_name
                                ? `http://127.0.0.1:8000/uploads/${item.image_name}`
                                : null;
                            };

                            const imageUrl = getResolvedImageUrl(row);
                            return imageUrl ? (
                              <a
                                href={imageUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="predictionHistoryImageLink"
                              >
                                {row.image_name || 'View Image'}
                              </a>
                            ) : (
                              'Image Not Available'
                            );
                          })()}
                        </td>
                        <td>
                    <span className="phBadge phBadgeDisease">{row.predicted_disease ?? 'Unknown'}</span>
                        </td>
                        <td className="phPrimaryText">{fmtConfidence(row.confidence_score)}</td>
                        <td className="phMetaText">{formatTimestamp(row.timestamp)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <div className="phPagination">
                <button 
                  className="phPageBtn"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  Previous
                </button>
                <span className="phPageInfo">
                  Page {currentPage} of {totalPages} ({filteredRows.length} total)
                </span>
                <button 
                  className="phPageBtn"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}

        {showPreview && selectedRecord && (
          <div className="phPreviewOverlay" onClick={handleClosePreview}>
            <div className="phPreviewPanel" onClick={(e) => e.stopPropagation()}>
              <div className="phPreviewHeader">
                <h3 className="phPreviewTitle">Image Preview</h3>
                <button className="phCloseBtn" onClick={handleClosePreview}>×</button>
              </div>
              <div className="phPreviewContent">
                <div className="phPreviewImage">
                  {selectedRecord.image_name ? (
                    <img 
                      src={selectedRecord.image_name} 
                      alt={selectedRecord.predicted_disease || 'Uploaded image'}
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                  ) : null}
                  <div className="phPlaceholder" style={{ display: selectedRecord.image_name ? 'none' : 'flex' }}>
                    <div className="phPlaceholderIcon">📷</div>
                    <div className="phPlaceholderText">{selectedRecord.image_name || 'No image available'}</div>
                  </div>
                </div>
                <div className="phPreviewDetails">
                  <div className="phDetailRow">
                    <span className="phDetailLabel">Image Name:</span>
                    <span className="phDetailValue phPrimaryText">{selectedRecord.image_name || 'N/A'}</span>
                  </div>
                  <div className="phDetailRow">
                    <span className="phDetailLabel">Predicted Disease:</span>
                    <span className="phDetailValue phPrimaryText">{selectedRecord.predicted_disease || 'Unknown'}</span>
                  </div>
                  <div className="phDetailRow">
                    <span className="phDetailLabel">Confidence Score:</span>
                    <span className="phDetailValue phPrimaryText">{fmtConfidence(selectedRecord.confidence_score)}</span>
                  </div>
                  <div className="phDetailRow">
                    <span className="phDetailLabel">Prediction Source:</span>
                    <span className="phDetailValue">
                      <span 
                        className="phBadge phBadgeSource"
                        style={{
                          background: getSourceBadgeColor(selectedRecord.prediction_source).bg,
                          borderColor: getSourceBadgeColor(selectedRecord.prediction_source).border,
                          color: getSourceBadgeColor(selectedRecord.prediction_source).text
                        }}
                      >
                        {selectedRecord.prediction_source || 'Unknown'}
                      </span>
                    </span>
                  </div>
                  <div className="phDetailRow">
                    <span className="phDetailLabel">Timestamp:</span>
                    <span className="phDetailValue phMetaText">{formatTimestamp(selectedRecord.timestamp)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .phPage{padding:20px;background:#f9fafb;min-height:100vh;}
        .phHeader{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:20px;}
        .phTitleWrap{display:flex;flex-direction:column;gap:4px;}
        .phTitle{font-weight:700;font-size:24px;color:#1F2937;}
        .phSubtitle{font-size:14px;color:#4B5563;font-weight:500;}
        
        .phControls{display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap;}
        .phControlGroup{display:flex;flex-direction:column;gap:6px;}
        .phLabel{font-size:13px;font-weight:600;color:#374151;}
        .phSelect{padding:10px 14px;border:1px solid #d1d5db;border-radius:8px;font-size:14px;color:#1F2937;background:#fff;min-width:200px;cursor:pointer;}
        .phSelect:focus{outline:2px solid #3b82f6;border-color:#3b82f6;}
        .phInput{padding:10px 14px;border:1px solid #d1d5db;border-radius:8px;font-size:14px;color:#1F2937;background:#fff;min-width:300px;}
        .phInput:focus{outline:2px solid #3b82f6;border-color:#3b82f6;}
        .phInput::placeholder{color:#9ca3af;}
        
        .phEmpty{padding:24px;border-radius:12px;border:1px solid #e5e7eb;background:#fff;color:#4B5563;font-weight:500;text-align:center;}
        
        .phTableWrap{overflow:auto;background:#fff;border-radius:12px;border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
        .phTable{width:100%;border-collapse:separate;border-spacing:0;}
        .phTable th{position:sticky;top:0;background:#f8fafc;color:#374151;font-size:13px;font-weight:600;text-align:left;padding:14px 16px;border-bottom:2px solid #e5e7eb;}
        .phTable td{padding:14px 16px;border-bottom:1px solid #e5e7eb;vertical-align:middle;}
        .phRow{cursor:pointer;transition:background-color 0.15s;}
        .phRow:hover{background-color:#f3f4f6;}
        
        .phBadge{display:inline-flex;align-items:center;padding:6px 12px;border-radius:20px;font-weight:600;font-size:13px;border:1px solid;}
        .phBadgeDisease{background:rgba(59,130,246,0.1);border-color:rgba(59,130,246,0.3);color:#1e40af;}
        .phBadgeSource{background:rgba(148,163,184,0.1);border-color:rgba(148,163,184,0.3);color:#374151;}
        
        .phPrimaryText{color:#1F2937;font-weight:600;font-size:14px;}
        .phSecondaryText{color:#374151;font-weight:500;font-size:14px;}
        .phMetaText{color:#4B5563;font-weight:500;font-size:13px;}
        
        .phPagination{display:flex;align-items:center;justify-content:center;gap:16px;margin-top:20px;padding:16px;background:#fff;border-radius:12px;border:1px solid #e5e7eb;}
        .phPageBtn{padding:10px 20px;border-radius:8px;border:1px solid #d1d5db;background:#fff;font-size:14px;font-weight:600;color:#1F2937;cursor:pointer;transition:all 0.15s;}
        .phPageBtn:hover:not(:disabled){background:#f3f4f6;border-color:#9ca3af;}
        .phPageBtn:disabled{opacity:0.5;cursor:not-allowed;}
        .phPageInfo{font-size:14px;color:#4B5563;font-weight:500;}
        
        .phPreviewOverlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:1000;padding:20px;}
        .phPreviewPanel{background:#fff;border-radius:16px;max-width:600px;width:100%;max-height:90vh;overflow-y:auto;box-shadow:0 20px 25px -5px rgba(0,0,0,0.1),0 10px 10px -5px rgba(0,0,0,0.04);}
        .phPreviewHeader{display:flex;align-items:center;justify-content:space-between;padding:20px;border-bottom:1px solid #e5e7eb;}
        .phPreviewTitle{font-size:18px;font-weight:700;color:#1F2937;margin:0;}
        .phCloseBtn{width:32px;height:32px;border-radius:8px;border:none;background:#f3f4f6;font-size:24px;font-weight:700;color:#6b7280;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.15s;}
        .phCloseBtn:hover{background:#e5e7eb;color:#1F2937;}
        .phPreviewContent{padding:20px;}
        .phPreviewImage{display:flex;align-items:center;justify-content:center;background:#f9fafb;border-radius:12px;border:1px solid #e5e7eb;padding:20px;margin-bottom:20px;min-height:200px;}
        .phPreviewImage img{max-width:100%;max-height:300px;border-radius:8px;object-fit:contain;}
        .phPlaceholder{flex-direction:column;align-items:center;gap:12px;color:#9ca3af;}
        .phPlaceholderIcon{font-size:48px;}
        .phPlaceholderText{font-size:14px;font-weight:500;}
        .phPreviewDetails{display:flex;flex-direction:column;gap:12px;}
        .phDetailRow{display:flex;justify-content:space-between;align-items:center;padding:12px;background:#f9fafb;border-radius:8px;}
        .phDetailLabel{font-size:13px;font-weight:600;color:#4B5563;}
        .phDetailValue{font-size:14px;font-weight:600;}
        
        .phCenter{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:200px;}
        .phSpinner{width:40px;height:40px;border-radius:50%;border:4px solid #e5e7eb;border-top-color:#3b82f6;animation:phspin 0.9s linear infinite;}
        @keyframes phspin{to{transform:rotate(360deg)}}
        .phMuted{color:#6b7280;font-weight:500;}
        .phErrorWrap{padding:24px;border-radius:16px;border:1px solid #fecaca;background:#fef2f2;text-align:center;}
        .phErrorTitle{color:#991b1b;font-weight:700;margin-bottom:8px;font-size:18px;}
        .phErrorMsg{color:#7f1d1d;font-weight:500;margin-bottom:16px;}
        .phBtn{border-radius:8px;padding:12px 24px;font-weight:600;font-size:14px;cursor:pointer;border:1px solid transparent;transition:all 0.15s;}
        .phBtnPrimary{background:#3b82f6;border-color:#2563eb;color:#fff;}
        .phBtnPrimary:hover{background:#2563eb;}

        .predictionHistoryImageLink{color:#1F2937;text-decoration:none;font-weight:500;}
        .predictionHistoryImageLink:hover{color:#111827;text-decoration:underline;}
      `}</style>
    </Layout>
  );
}

