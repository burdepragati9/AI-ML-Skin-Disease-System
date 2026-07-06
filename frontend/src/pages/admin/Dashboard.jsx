import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

import Layout from '../../components/layout/Layout';
import api from '../../services/api';

import './Dashboard.css';

const formatPct = (v) => `${v}%`;

const rechartsTheme = {
  axisText: '#94a3b8',
  grid: 'rgba(148, 163, 184, 0.16)',
  tooltipBg: 'rgba(2, 6, 23, 0.92)',
  tooltipText: '#e5e7eb',
};

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
      <div style={{ color: rechartsTheme.tooltipText, fontWeight: 800, marginBottom: 6, fontSize: 13 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: rechartsTheme.tooltipText, fontSize: 13, opacity: 0.95 }}>
          <span style={{ color: p.fill || rechartsTheme.tooltipText, fontWeight: 800, marginRight: 8 }}>{p.name}:</span>
          {p.value}
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [analyticsData, setAnalyticsData] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAnalyticsData() {
      try {
        const response = await api.get('/admin/analytics');
        const data = response?.data;
        console.log("Admin Analytics Response:", data);
        if (!cancelled) {
          setAnalyticsData(data);
        }
      } catch (e) {
        console.error('Failed to load admin analytics:', e);
        if (!cancelled) {
          setError('Failed to load analytics data');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadAnalyticsData();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <Layout title="Admin Dashboard" userLabel="Admin">
        <div className="adminDashboard">
          <div className="adminDashboardHeader">
            <div className="adminDashboardTitleWrap">
              <div className="adminDashboardTitle">Admin Analytics</div>
              <div className="adminDashboardSubtitle">Loading...</div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Admin Dashboard" userLabel="Admin">
        <div className="adminDashboard">
          <div className="adminDashboardHeader">
            <div className="adminDashboardTitleWrap">
              <div className="adminDashboardTitle">Admin Analytics</div>
              <div className="adminDashboardSubtitle">Error</div>
            </div>
          </div>
          <div className="adminDashboardError">{error}</div>
        </div>
      </Layout>
    );
  }

  const diseaseFrequency = analyticsData?.disease_counts || [];
  const predictionSources = analyticsData?.source_counts || [];
  const aiRecognizedImages = analyticsData?.ai_recognized_images || [];

  return (
    <Layout title="Admin Dashboard" userLabel="Admin">
      <div className="adminDashboard">
        <header className="adminDashboardHeader">
          <div className="adminDashboardTitleWrap">
            <div className="adminDashboardTitle">Admin Analytics</div>
            <div className="adminDashboardSubtitle">Admin analytics overview</div>
          </div>
        </header>

        <section className="adminDashboardCards" aria-label="Summary statistics">
          <div className="adminAnalyticsCard adminAnalyticsCardBlue">
            <div className="adminAnalyticsIcon">🧠</div>
            <div className="adminAnalyticsContent">
              <div className="adminAnalyticsValue">{analyticsData?.ai_images || 0}</div>
              <div className="adminAnalyticsLabel">AI Images</div>
              <div className="adminAnalyticsDesc">Total recognized</div>
            </div>
          </div>
          <div className="adminAnalyticsCard adminAnalyticsCardGreen">
            <div className="adminAnalyticsIcon">🔄</div>
            <div className="adminAnalyticsContent">
              <div className="adminAnalyticsValue">{analyticsData?.retrained_images || 0}</div>
              <div className="adminAnalyticsLabel">Retrained Images</div>
              <div className="adminAnalyticsDesc">Model updates</div>
            </div>
          </div>
          <div className="adminAnalyticsCard adminAnalyticsCardPurple">
            <div className="adminAnalyticsIcon">🦠</div>
            <div className="adminAnalyticsContent">
              <div className="adminAnalyticsValue">{analyticsData?.most_added_disease || 'None'}</div>
              <div className="adminAnalyticsLabel">Most Added Disease</div>
              <div className="adminAnalyticsDesc">Common prediction</div>
            </div>
          </div>
          <div className="adminAnalyticsCard adminAnalyticsCardTeal">
            <div className="adminAnalyticsIcon">📈</div>
            <div className="adminAnalyticsContent">
              <div className="adminAnalyticsValue">{analyticsData?.accuracy_improvement || '0%'}</div>
              <div className="adminAnalyticsLabel">Accuracy Improvement</div>
              <div className="adminAnalyticsDesc">Model performance</div>
            </div>
          </div>
        </section>

        <section className="adminDashboardGrid" aria-label="Analytics charts">
          <div className="sectionCard">
            <div className="sectionHeader">
              <div className="sectionTitle">Disease Frequency</div>
              <div className="sectionHint">Bar chart</div>
            </div>

            <div style={{ width: '100%', height: 320, minHeight: 320 }}>
              <ResponsiveContainer>
                <BarChart data={diseaseFrequency} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
                  <XAxis dataKey="disease" stroke={rechartsTheme.axisText} tick={{ fill: rechartsTheme.axisText, fontSize: 12 }} />

                  <YAxis stroke={rechartsTheme.axisText} tick={{ fill: rechartsTheme.axisText, fontSize: 12 }} />
                  <Tooltip content={<MedicalTooltip />} />
                  <Bar dataKey="count" name="Cases" fill="rgba(59, 130, 246, 0.85)" radius={[10, 10, 4, 4]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="sectionCard">
            <div className="sectionHeader">
              <div className="sectionTitle">Prediction Sources</div>
              <div className="sectionHint">Pie chart</div>
            </div>

            <div style={{ width: '100%', height: 320, minHeight: 320 }}>
              <ResponsiveContainer>
                <PieChart>

                  <Tooltip
                    formatter={(value) => formatPct(value)}
                    content={(props) => {
                      const { active, payload } = props;
                      if (!active || !payload?.length) return null;
                      const item = payload[0];
                      return (
                        <MedicalTooltip active={active} payload={payload} label={item.name} />
                      );
                    }}
                  />
                  <Legend />
                  <Pie
                    data={predictionSources}
                    dataKey="count"
                    nameKey="prediction_source"
                    innerRadius={70}
                    outerRadius={110}
                    paddingAngle={3}
                  >
                    {predictionSources.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? "rgba(99, 102, 241, 0.95)" : "rgba(16, 185, 129, 0.95)"} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        <section className="sectionCard" aria-label="AI recognized images">
          <div className="sectionHeader">
            <div className="sectionTitle">AI Recognized Images</div>
            <div className="sectionHint">Latest inferences</div>
          </div>

          <div className="tableWrap">
            <table className="adminTable">
              <thead>
                <tr>
                  <th>Image Name</th>
                  <th>Disease</th>
                  <th>Confidence</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {aiRecognizedImages.map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ color: '#1F2937', fontWeight: 600 }}>{row.image_name || '-'}</td>
                    <td>
                      <span className="badge">{row.predicted_disease || '-'}</span>
                    </td>
                    <td style={{ color: '#111827', fontWeight: 700 }}>{row.confidence ? `${(row.confidence * 100).toFixed(2)}%` : '-'}</td>
                    <td style={{ color: '#374151' }}>{row.created_at ? new Date(row.created_at).toLocaleDateString() : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </Layout>
  );
}


