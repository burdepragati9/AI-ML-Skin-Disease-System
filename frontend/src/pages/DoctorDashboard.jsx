import { useEffect, useMemo, useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts';

import Layout from '../components/layout/Layout';
import api from '../services/api';
import './DoctorDashboard.css';

// Module-level singleton promise so React StrictMode's double effect fire
// (and any concurrent mounts) result in a single /dashboard/doctor request.
let dashboardRequestPromise = null;
const loadDoctorDashboard = () => {
  if (!dashboardRequestPromise) {
    dashboardRequestPromise = api.get('/dashboard/doctor').finally(() => {
      dashboardRequestPromise = null;
    });
  }
  return dashboardRequestPromise;
};

const rechartsTheme = {
  axisText: '#94a3b8',
  grid: 'rgba(148, 163, 184, 0.16)',
  tooltipBg: 'rgba(2, 6, 23, 0.92)',
  tooltipText: '#e5e7eb',
};

const getImageUrl = (imagePath) => {
  if (!imagePath) return null;

  // Convert stored filesystem path to public URL
  if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
    return imagePath;
  }

  // Handle absolute Windows paths like: C:\Updated_Project\history\uploads\Vitiligo\search_Vitiligo_37f1803263.jpg
  const normalizedPath = imagePath.replace(/\\/g, '/');
  
  // Extract the part after history/uploads/
  const marker = 'history/uploads/';
  const idx = normalizedPath.indexOf(marker);
  
  if (idx !== -1) {
    // Get everything after history/uploads/ including disease subdirectory
    const relPart = normalizedPath.substring(idx + marker.length);
    return `http://127.0.0.1:8000/uploads/${relPart}`;
  }

  // Handle relative paths starting with uploads/
  if (normalizedPath.startsWith('uploads/')) {
    return `http://127.0.0.1:8000/${normalizedPath}`;
  }

  return null;
};

function LoadingState() {
  return (
    <div className="bbCenter" aria-label="Loading">
      <div className="bbSpinner" />
      <div className="bbMuted" style={{ marginTop: 12 }}>
        Loading doctor dashboard...
      </div>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="bbErrorWrap" aria-label="Error">
      <div className="bbErrorTitle">Failed to load dashboard</div>
      <div className="bbErrorMsg">{message}</div>
      {onRetry && (
        <button type="button" className="bbBtn bbBtnPrimary" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

function MedicalTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div
      className="bbTooltip"
      style={{
        background: rechartsTheme.tooltipBg,
        border: '1px solid rgba(148, 163, 184, 0.18)',
        borderRadius: 10,
        padding: '10px 12px',
        boxShadow: '0 18px 40px rgba(2, 6, 23, 0.5)',
      }}
    >
      <div
        style={{
          color: rechartsTheme.tooltipText,
          fontWeight: 800,
          marginBottom: 6,
          fontSize: 13,
        }}
      >
        {label}
      </div>
      {payload.map((p) => (
        <div
          key={p.name}
          style={{
            color: rechartsTheme.tooltipText,
            fontSize: 13,
            opacity: 0.95,
          }}
        >
          <span
            style={{
              color: p.color || p.fill || rechartsTheme.tooltipText,
              fontWeight: 800,
              marginRight: 8,
            }}
          >
            {p.name}:
          </span>
          {p.value}
        </div>
      ))}
    </div>
  );
}

function formatCountAxisTick(v) {
  if (!Number.isFinite(v)) return '';
  return `${v}`;
}

export default function DoctorDashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [stats, setStats] = useState(null);
  const [diseases, setDiseases] = useState([]);
  const [sources, setSources] = useState([]);
  const [images, setImages] = useState([]);
  const [history, setHistory] = useState({ rows: [] });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError('');

      try {
        const response = await loadDoctorDashboard();
        if (cancelled) return;

        const data = response.data || {};
        setStats(data);
        setDiseases(Array.isArray(data.most_searched_diseases) ? data.most_searched_diseases : []);
        setSources(Array.isArray(data.ai_vs_ml) ? data.ai_vs_ml : []);
        setImages(Array.isArray(data.most_searched_images) ? data.most_searched_images : []);
        setHistory({ rows: Array.isArray(data.prediction_history) ? data.prediction_history : [] });
      } catch (e) {
        if (cancelled) return;
        const msg = e?.response?.data?.detail || e?.message || 'Unexpected error';
        setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const diseaseChartData = useMemo(() => diseases, [diseases]);
  const predictionSourceChartData = useMemo(() => sources, [sources]);

  const pieData = useMemo(() => {
    return (predictionSourceChartData || []).map((s) => ({
      name: s.prediction_source,
      value: s.count,
    }));
  }, [predictionSourceChartData]);

  const pieCells = ['rgba(99, 102, 241, 0.95)', 'rgba(16, 185, 129, 0.95)', 'rgba(236, 72, 153, 0.95)'];

  const totalSearches = stats?.total_searches ?? 0;
  const freeSearchesLeft = stats?.free_searches_left ?? 0;
  const usedSearches = stats?.used_searches ?? 0;

  // Calculate average confidence from prediction history
  const avgConfidence = useMemo(() => {
    const historyData = history?.rows || [];
    if (historyData.length === 0) return 0;

    // IMPORTANT: Backend returns confidence as `confidence_score` (see doctor_dashboard.py)
    const values = historyData
      .map((r) => r?.confidence_score)
      .filter((v) => v !== null && v !== undefined);

    if (values.length === 0) return null;

    const sum = values.reduce((acc, v) => acc + Number(v), 0);
    return (sum / values.length).toFixed(1);
  }, [history]);


  if (loading) {
    return (
      <Layout title="Doctor Dashboard" userLabel="Doctor" role="doctor">
        <LoadingState />
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Doctor Dashboard" userLabel="Doctor" role="doctor">
        <ErrorState message={error} onRetry={() => window.location.reload()} />
      </Layout>
    );
  }

  return (
    <Layout title="Doctor Dashboard" userLabel="Doctor" role="doctor">
      <div className="doctorDashboard">
        <style>{`
          .dashboardImageLink{color:#1F2937;text-decoration:none;font-weight:500;}
          .dashboardImageLink:hover{color:#111827;text-decoration:underline;}
        `}</style>
        <header className="doctorDashboardHeader">

          <div className="doctorDashboardTitleWrap">
            <div className="doctorDashboardTitle">Doctor Dashboard</div>
            <div className="doctorDashboardSubtitle">Your clinical activity analytics</div>
          </div>

          <div className="doctorDashboardBadges">
            {freeSearchesLeft <= 0 ? (
              <div className="doctorBadge doctorBadge--warning">Free limit reached</div>
            ) : (
              <div className="doctorBadge doctorBadge--ok">Free searches available</div>
            )}
          </div>
        </header>

        <section className="doctorDashboardCards" aria-label="Summary statistics">
          <div className="ddAnalyticsCard ddAnalyticsCardBlue">
            <div className="ddAnalyticsIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.35-4.35" />
              </svg>
            </div>
            <div className="ddAnalyticsContent">
              <div className="ddAnalyticsValue">{totalSearches}</div>
              <div className="ddAnalyticsLabel">Total Searches</div>
              <div className="ddAnalyticsDesc">All predictions made</div>
            </div>
          </div>

          <div className="ddAnalyticsCard ddAnalyticsCardGreen">
            <div className="ddAnalyticsIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
              </svg>
            </div>
            <div className="ddAnalyticsContent">
              <div className="ddAnalyticsValue">{usedSearches}</div>
              <div className="ddAnalyticsLabel">Used Searches</div>
              <div className="ddAnalyticsDesc">Predictions consumed</div>
            </div>
          </div>

          <div className="ddAnalyticsCard ddAnalyticsCardPurple">
            <div className="ddAnalyticsIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 12V8H6a2 2 0 0 1-2-2c0-1.1.9-2 2-2h12v4" />
                <path d="M4 6v12c0 1.1.9 2 2 2h14v-4" />
                <path d="M18 12a2 2 0 0 0-2 2c0 1.1.9 2 2 2h4v-4h-4z" />
              </svg>
            </div>
            <div className="ddAnalyticsContent">
              <div className="ddAnalyticsValue">{freeSearchesLeft}</div>
              <div className="ddAnalyticsLabel">Free Searches Left</div>
              <div className="ddAnalyticsDesc">Remaining quota</div>
            </div>
          </div>

          <div className="ddAnalyticsCard ddAnalyticsCardOrange">
            <div className="ddAnalyticsIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            </div>
            <div className="ddAnalyticsContent">
              <div className="ddAnalyticsValue">{avgConfidence}%</div>
              <div className="ddAnalyticsLabel">Average Confidence</div>
              <div className="ddAnalyticsDesc">AI prediction accuracy</div>
            </div>
          </div>
        </section>

        <section className="doctorDashboardGrid" aria-label="Charts">
          <div className="sectionCard ddChartCard">
            <div className="sectionHeader">
              <div className="sectionTitle">Top Searched Diseases</div>
              <div className="sectionHint">Most frequently detected conditions</div>
            </div>

            <div className="bbChartWrap" style={{ width: '100%', minHeight: 300 }}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={diseaseChartData}>
                  <XAxis dataKey="disease" stroke={rechartsTheme.axisText} tick={{ fill: rechartsTheme.axisText, fontSize: 12 }} />
                  <YAxis stroke={rechartsTheme.axisText} tick={{ fill: rechartsTheme.axisText, fontSize: 12 }} tickFormatter={formatCountAxisTick} />
                  <Tooltip content={<MedicalTooltip />} />
                  <Bar dataKey="count" name="Cases" fill="url(#barGradient)" radius={[8, 8, 0, 0]}>
                    <defs>
                      <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#2563EB" stopOpacity={0.9} />
                        <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.6} />
                      </linearGradient>
                    </defs>
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {(!diseaseChartData || diseaseChartData.length === 0) && <div className="bbEmpty">No disease data yet.</div>}
          </div>

          <div className="sectionCard ddChartCard">
            <div className="sectionHeader">
              <div className="sectionTitle">Prediction Source Distribution</div>
              <div className="sectionHint">AI model vs ML predictions</div>
            </div>

            <div className="bbChartWrap bbChartWrap--pie" style={{ width: '100%', minHeight: 300 }}>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Tooltip
                    formatter={(value) => `${value}`}
                    content={(props) => {
                      const { active, payload, label } = props;
                      if (!active || !payload?.length) return null;
                      const item = payload[0];
                      return <MedicalTooltip active={active} payload={payload} label={label || item?.name} />;
                    }}
                  />
                  <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={60} outerRadius={100} paddingAngle={5}>
                    {pieData.map((_, idx) => (
                      <Cell key={`cell-${idx}`} fill={pieCells[idx % pieCells.length]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="ddLegendContainer">
              {pieData.map((item, idx) => (
                <div key={item.name} className="ddLegendItem">
                  <div className="ddLegendColor" style={{ backgroundColor: pieCells[idx % pieCells.length] }} />
                  <span className="ddLegendLabel">{item.name}</span>
                  <span className="ddLegendValue">{item.value}</span>
                </div>
              ))}
            </div>

            {(!pieData || pieData.length === 0) && <div className="bbEmpty">No prediction source data yet.</div>}
          </div>
        </section>

        <section className="sectionCard" aria-label="Prediction History">
          <div className="sectionHeader">
            <div className="sectionTitle">Recent Predictions</div>
            <div className="sectionHint">Latest AI-powered diagnoses</div>
          </div>

          {(!history?.rows || history.rows.length === 0) && <div className="bbEmpty">No search history found.</div>}

          {history?.rows?.length > 0 && (
            <div className="ddTableContainer">
              <table className="ddPredictionTable" aria-label="Prediction history table">
                <thead>
                  <tr>
                    <th>Image</th>
                    <th>Disease</th>
                    <th>Confidence</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {history.rows.map((r, idx) => {
                    const item = r;

                    const API_BASE_URL = 'http://127.0.0.1:8000';
                    const imageUrlFromBackend = item?.image_url;
                    const imagePathFromBackend = item?.image_path;

                    const isHttpUrl = (v) => typeof v === 'string' && (v.startsWith('http://') || v.startsWith('https://'));

                    // Use getImageUrl to convert absolute paths to public URLs
                    const imageUrlFromPath = getImageUrl(imagePathFromBackend);
                    
                    // Fallback to image_name if no path available
                    const imageNameFallbackUrl = item?.image_name
                      ? `${API_BASE_URL}/uploads/${item.image_name}`
                      : '';

                    const imageSrc =
                      isHttpUrl(imageUrlFromBackend)
                        ? imageUrlFromBackend
                        : imageUrlFromPath
                          ? imageUrlFromPath
                          : imageNameFallbackUrl;


                    const disease = item?.predicted_disease ?? item?.disease;
                    const confidenceRaw = item?.confidence_score ?? item?.confidence;
                    const createdAt = item?.timestamp ?? item?.created_at;

                    const confidence = Number(confidenceRaw);
                    let confidenceClass = 'ddConfidenceLow';
                    if (confidence >= 80) confidenceClass = 'ddConfidenceHigh';
                    else if (confidence >= 60) confidenceClass = 'ddConfidenceMedium';

                    return (
                      <tr key={`${createdAt || idx}-${idx}`}>
                        <td>
                          {imageSrc ? (
                              <a
                              href={imageSrc}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="ddImageLink dashboardImageLink"
                              title={imageSrc}
                            >
                              {item?.image_name || 'View Image'}
                            </a>
                          ) : (
                            <span className="ddMutedText">Image Not Available</span>
                          )}
                        </td>
                        <td>
                          <span className="ddDiseaseBadge">{disease ?? 'N/A'}</span>
                        </td>
                        <td>
                          <span className={`ddConfidenceBadge ${confidenceClass}`}>
                            {Number.isFinite(confidence) ? `${confidence.toFixed(1)}%` : 'N/A'}
                          </span>
                        </td>
                        <td className="ddDateCell">{createdAt ?? 'N/A'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>


        <section className="sectionCard" aria-label="Disease Statistics">
          <div className="sectionHeader">
            <div className="sectionTitle">Disease Statistics</div>
            <div className="sectionHint">Case distribution by condition</div>
          </div>

          {(!diseases || diseases.length === 0) && <div className="bbEmpty">No disease statistics available.</div>}

          {diseases?.length > 0 && (
            <div className="ddStatsContainer">
              {diseases.map((disease, idx) => {
                const maxCount = Math.max(...diseases.map((d) => d.count || 0));
                const percentage = maxCount > 0 ? ((disease.count || 0) / maxCount) * 100 : 0;
                const colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];
                const color = colors[idx % colors.length];

                return (
                  <div key={disease.disease} className="ddStatItem">
                    <div className="ddStatHeader">
                      <span className="ddStatName">{disease.disease}</span>
                      <span className="ddStatCount">{disease.count} cases</span>
                    </div>
                    <div className="ddProgressBar">
                      <div className="ddProgressFill" style={{ width: `${percentage}%`, backgroundColor: color }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="ddInsightsCard" aria-label="AI Insights">
          <div className="ddInsightsHeader">
            <div className="ddInsightsIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" />
                <path d="M12 6v6l4 2" />
              </svg>
            </div>
            <div>
              <div className="ddInsightsTitle">AI Clinical Insights</div>
              <div className="ddInsightsSubtitle">Powered by your prediction data</div>
            </div>
          </div>
            <div className="ddInsightsGrid">
            <div className="ddInsightItem">
              <div className="ddInsightLabel">Most Detected Disease</div>
              <div className="ddInsightValue">{diseases?.[0]?.disease || 'N/A'}</div>
            </div>
            <div className="ddInsightItem">
              <div className="ddInsightLabel">Average Confidence</div>
              <div className="ddInsightValue">{avgConfidence}%</div>
            </div>
            <div className="ddInsightItem">
              <div className="ddInsightLabel">Total Predictions</div>
              <div className="ddInsightValue">{totalSearches}</div>
            </div>
          </div>
        </section>

        {history?.rows?.length > 0 && (
          <div className="bbFootNote">Tip: Your dashboard uses real-time data from your prediction history.</div>
        )}
      </div>
    </Layout>
  );
}

