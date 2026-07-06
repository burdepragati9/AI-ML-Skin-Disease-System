import { useEffect, useState } from 'react';
import Layout from '../../components/layout/Layout';
import api from '../../services/api';
import './SystemMonitoring.css';

export default function SystemMonitoring() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingLogs, setTrainingLogs] = useState([]);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        const [summaryRes, statusRes, logsRes] = await Promise.all([
          api.get('/admin/analytics'),
          api.get('/admin/training-status'),
          api.get('/admin/training-logs')
        ]);

        if (!cancelled) {
          setSummary(summaryRes?.data || {});
          setTrainingStatus(statusRes?.data || {});
          setTrainingLogs(logsRes?.data || []);
        }
      } catch (e) {
        console.error('Failed to load system monitoring data:', e);
        if (!cancelled) {
          setError('Failed to load system monitoring data');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <Layout title="System Monitoring" userLabel="Admin">
        <div className="systemMonitoring">
          <div className="systemMonitoringHeader">
            <div className="systemMonitoringTitleWrap">
              <div className="systemMonitoringTitle">System Monitoring</div>
              <div className="systemMonitoringSubtitle">Loading...</div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="System Monitoring" userLabel="Admin">
        <div className="systemMonitoring">
          <div className="systemMonitoringHeader">
            <div className="systemMonitoringTitleWrap">
              <div className="systemMonitoringTitle">System Monitoring</div>
              <div className="systemMonitoringSubtitle">Error</div>
            </div>
          </div>
          <div className="systemMonitoringError">{error}</div>
        </div>
      </Layout>
    );
  }

  const getStatusBadgeColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
      case 'success':
        return 'statusBadge--success';
      case 'processing':
      case 'in_progress':
        return 'statusBadge--processing';
      case 'failed':
      case 'error':
        return 'statusBadge--failed';
      case 'queued':
      default:
        return 'statusBadge--queued';
    }
  };

  const getImageUrl = (imagePath) => {
    if (!imagePath) return null;

    // If already a URL, return as-is
    if (typeof imagePath === 'string' && (imagePath.startsWith('http://') || imagePath.startsWith('https://'))) {
      return imagePath;
    }

    // We intentionally do not hardcode the FastAPI origin.
    // Instead, we reuse the current backend base if available.
    const backendBase = api?.defaults?.baseURL || '';

    // Normalize windows paths -> unix-like
    const normalized = String(imagePath).replace(/\\/g, '/');

    // Expected format: C:/Updated_Project/history/uploads/Disease/search_xxx.jpg
    const marker = 'history/uploads/';
    const idx = normalized.indexOf(marker);

    if (idx !== -1) {
      const relPart = normalized.substring(idx + marker.length);
      const base = backendBase.replace(/\/+$/, '');
      return `${base}/uploads/${relPart}`;
    }

    // Fallback for relative paths
    const rel = normalized.replace(/^\/+/, '');
    if (rel.startsWith('uploads/')) {
      const base = backendBase.replace(/\/+$/, '');
      return `${base}/${rel}`;
    }

    return null;
  };

  const lastLog = trainingStatus?.last_log;

  return (
    <Layout title="System Monitoring" userLabel="Admin">
      <div className="systemMonitoring">
        <header className="systemMonitoringHeader">
          <div className="systemMonitoringTitleWrap">
            <div className="systemMonitoringTitle">System Monitoring</div>
            <div className="systemMonitoringSubtitle">System health and training status</div>
          </div>
        </header>

        {/* Summary Cards */}
        <section className="systemMonitoringCards">
          <div className="systemMonitoringCard systemMonitoringCardBlue">
            <div className="systemMonitoringCardIcon">🧠</div>
            <div className="systemMonitoringCardContent">
              <div className="systemMonitoringCardValue">{summary?.ai_images || 0}</div>
              <div className="systemMonitoringCardLabel">AI Images</div>
            </div>
          </div>
          <div className="systemMonitoringCard systemMonitoringCardGreen">
            <div className="systemMonitoringCardIcon">🔄</div>
            <div className="systemMonitoringCardContent">
              <div className="systemMonitoringCardValue">{summary?.retrained_images || 0}</div>
              <div className="systemMonitoringCardLabel">Retrained Images</div>
            </div>
          </div>
          <div className="systemMonitoringCard systemMonitoringCardPurple">
            <div className="systemMonitoringCardIcon">🦠</div>
            <div className="systemMonitoringCardContent">
              <div className="systemMonitoringCardValue">{summary?.most_added_disease || 'None'}</div>
              <div className="systemMonitoringCardLabel">Most Added Disease</div>
            </div>
          </div>
          <div className="systemMonitoringCard systemMonitoringCardTeal">
            <div className="systemMonitoringCardIcon">📈</div>
            <div className="systemMonitoringCardContent">
              <div className="systemMonitoringCardValue">{summary?.accuracy_improvement || '0%'}</div>
              <div className="systemMonitoringCardLabel">Accuracy Improvement</div>
            </div>
          </div>
        </section>

        {/* Training Queue Status */}
        <section className="systemMonitoringSection">
          <div className="sectionHeader">
            <div className="sectionTitle">Training Queue Status</div>
          </div>
          <div className="statusCards">
            <div className="statusCard statusCard--queued">
              <div className="statusCardValue">{trainingStatus?.queued || 0}</div>
              <div className="statusCardLabel">Queued</div>
            </div>
            <div className="statusCard statusCard--processing">
              <div className="statusCardValue">{trainingStatus?.processing || 0}</div>
              <div className="statusCardLabel">Processing</div>
            </div>
            <div className="statusCard statusCard--completed">
              <div className="statusCardValue">{trainingStatus?.completed || 0}</div>
              <div className="statusCardLabel">Completed</div>
            </div>
            <div className="statusCard statusCard--failed">
              <div className="statusCardValue">{trainingStatus?.failed || 0}</div>
              <div className="statusCardLabel">Failed</div>
            </div>
          </div>
        </section>

        {/* Latest Training Activity */}
        {lastLog && (
          <section className="systemMonitoringSection">
            <div className="sectionHeader">
              <div className="sectionTitle">Latest Training Activity</div>
            </div>
            <div className="activityCard">
              <div className="activityGrid">
                <div className="activityItem">
                  <div className="activityLabel">Training ID</div>
                  <div className="activityValue">{lastLog.id || '-'}</div>
                </div>
                <div className="activityItem">
                  <div className="activityLabel">Event Type</div>
                  <div className="activityValue">{lastLog.event_type || '-'}</div>
                </div>
                <div className="activityItem">
                  <div className="activityLabel">Status</div>
                  <span className={`statusBadge ${getStatusBadgeColor(lastLog.status)}`}>
                    {lastLog.status || 'Unknown'}
                  </span>
                </div>
                <div className="activityItem">
                  <div className="activityLabel">Disease</div>
                  <div className="activityValue">{lastLog.disease || '-'}</div>
                </div>
                <div className="activityItem">
                  <div className="activityLabel">Accuracy Before</div>
                  <div className="activityValue">
                    {lastLog.accuracy_before ? `${(lastLog.accuracy_before * 100).toFixed(2)}%` : '-'}
                  </div>
                </div>
                <div className="activityItem">
                  <div className="activityLabel">Accuracy After</div>
                  <div className="activityValue">
                    {lastLog.accuracy_after ? `${(lastLog.accuracy_after * 100).toFixed(2)}%` : '-'}
                  </div>
                </div>
                <div className="activityItem">
                  <div className="activityLabel">Date</div>
                  <div className="activityValue">
                    {lastLog.created_at ? new Date(lastLog.created_at).toLocaleString() : '-'}
                  </div>
                </div>
                <div className="activityItem activityItem--full">
                  <div className="activityLabel">Message</div>
                  <div className="activityValue">{lastLog.message || '-'}</div>
                </div>
              </div>

              {/* Training Image Preview */}
              {lastLog.image_path && getImageUrl(lastLog.image_path) && (
                <div className="activityImage">
                  <div className="activityLabel">Training Image</div>
                  <div className="activityImagePreview">
                    <img
                      src={getImageUrl(lastLog.image_path)}
                      alt="Training"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        if (e.target.nextSibling) e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                    <div className="activityImageError" style={{ display: 'none' }}>
                      Preview unavailable
                    </div>
                  </div>
                </div>
              )}

            </div>
          </section>
        )}

        {/* Recent Training Logs */}
        <section className="systemMonitoringSection">
          <div className="sectionHeader">
            <div className="sectionTitle">Recent Training Logs</div>
          </div>
          <div className="tableWrap">
            <table className="adminTable">
              <thead>
                <tr>
                  <th>Training ID</th>
                  <th>Disease</th>
                  <th>Status</th>
                  <th>Date</th>
                  <th>Message</th>
                </tr>
              </thead>
              <tbody>
                {trainingLogs.length > 0 ? (
                  trainingLogs.map((log) => (
                    <tr key={log.id}>
                      <td style={{ color: '#1F2937', fontWeight: 600 }}>{log.id}</td>
                      <td style={{ color: '#374151' }}>{log.disease || '-'}</td>
                      <td>
                        <span className={`statusBadge ${getStatusBadgeColor(log.status)}`}>
                          {log.status || 'Unknown'}
                        </span>
                      </td>
                      <td style={{ color: '#374151' }}>
                        {log.created_at ? new Date(log.created_at).toLocaleDateString() : '-'}
                      </td>
                      <td style={{ color: '#374151' }}>{log.message || '-'}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', padding: '32px', color: '#6B7280' }}>
                      No training logs found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* System Health */}
        <section className="systemMonitoringSection systemMonitoringSection--health">
          <div className="sectionHeader sectionHeader--tight">
            <div className="sectionTitle">System Health</div>
          </div>
          <div className="healthBadges">
            <div className="healthBadge healthBadge--success">
              <span className="healthBadgeDot"></span>
              Backend Online
            </div>
            <div className="healthBadge healthBadge--success">
              <span className="healthBadgeDot"></span>
              Database Connected
            </div>
            <div className="healthBadge healthBadge--success">
              <span className="healthBadgeDot"></span>
              Model Loaded
            </div>
            <div className="healthBadge healthBadge--success">
              <span className="healthBadgeDot"></span>
              API Running
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
