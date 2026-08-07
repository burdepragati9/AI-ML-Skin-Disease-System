import { useEffect, useState } from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import Layout from '../components/layout/Layout';
import api from '../services/api';
import jsPDF from 'jspdf';
import './Reports.css';

export default function Reports() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const [dashboardData, setDashboardData] = useState(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadReportsData() {
      setLoading(true);
      setError('');
      
      try {
        // Load data from existing /dashboard/doctor endpoint
        const response = await api.get('/dashboard/doctor');
        const analyticsData = response?.data;
        console.log("Doctor Analytics Response:", analyticsData);

        if (!cancelled) {
          setDashboardData(analyticsData);
        }
      } catch (e) {
        if (!cancelled) {
          const msg = e?.response?.data?.detail || e?.message || 'Failed to load reports data';
          setError(msg);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadReportsData();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleDownloadPdf = async () => {
    setDownloadingPdf(true);
    try {
      console.log('Starting PDF generation...');
      console.log('avgConfidence:', avgConfidence);
      console.log('typeof avgConfidence:', typeof avgConfidence);
      
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 20;
      let y = margin;

      // Color theme
      const colors = {
        blue: [37, 99, 235],      // #2563EB
        lightBlue: [219, 234, 254], // #DBEAFE
        green: [16, 185, 129],    // #10B981
        purple: [139, 92, 246],   // #8B5CF6
        lightGray: [248, 250, 252], // #F8FAFC
        darkGray: [51, 65, 85],   // #334155
        white: [255, 255, 255]
      };

      // ==================== HEADER ====================
      // Blue header background
      doc.setFillColor(...colors.blue);
      doc.rect(0, 0, pageWidth, 50, 'F');
      
      // White text for header
      doc.setTextColor(...colors.white);
      doc.setFontSize(24);
      doc.setFont('helvetica', 'bold');
      doc.text('Skin Disease Detection System', pageWidth / 2, 20, { align: 'center' });
      
      doc.setFontSize(16);
      doc.setFont('helvetica', 'normal');
      doc.text('Doctor Analytics Report', pageWidth / 2, 32, { align: 'center' });
      
      doc.setFontSize(10);
      doc.text(`Generated: ${reportDate}`, pageWidth / 2, 42, { align: 'center' });
      
      y = 60;

      // ==================== DOCTOR INFORMATION CARD ====================
      doc.setFillColor(...colors.lightGray);
      doc.roundedRect(margin, y, pageWidth - (margin * 2), 35, 3, 3, 'F');
      
      doc.setTextColor(...colors.darkGray);
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('Doctor Information', margin + 5, y + 8);
      
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.text(`Name: ${doctorInfo?.full_name || '-'}`, margin + 5, y + 18);
      doc.text(`Specialization: ${doctorInfo?.specialization || '-'}`, margin + 5, y + 28);
      doc.text(`Report Date: ${reportDate}`, pageWidth - margin - 5, y + 18, { align: 'right' });
      
      y += 45;

      // ==================== CLINICAL ACTIVITY SUMMARY (4 KPI CARDS) ====================
      doc.setTextColor(...colors.darkGray);
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text('Clinical Activity Summary', margin, y);
      y += 10;

      const cardWidth = (pageWidth - (margin * 2) - 15) / 4;
      const cardHeight = 40;
      const cardColors = [colors.blue, colors.green, colors.purple, colors.blue];
      
      const kpiData = [
        { label: 'Total Cases', value: totalCases, color: colors.blue },
        { label: 'Total Searches', value: totalSearches, color: colors.green },
        { label: 'Highest Confidence', value: `${highestConfidence}%`, color: colors.purple },
        { label: 'Avg Confidence', value: `${avgConfidence}%`, color: colors.blue }
      ];

      kpiData.forEach((kpi, idx) => {
        const x = margin + (idx * (cardWidth + 5));
        
        // Card background
        doc.setFillColor(...colors.lightBlue);
        doc.roundedRect(x, y, cardWidth, cardHeight, 3, 3, 'F');
        
        // Colored accent bar
        doc.setFillColor(...kpi.color);
        doc.rect(x, y, cardWidth, 5, 'F');
        
        // Label
        doc.setTextColor(...colors.darkGray);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        doc.text(kpi.label, x + cardWidth / 2, y + 15, { align: 'center' });
        
        // Value
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.text(String(kpi.value), x + cardWidth / 2, y + 28, { align: 'center' });
      });
      
      y += cardHeight + 15;

      // ==================== DISEASE DISTRIBUTION TABLE ====================
      doc.setTextColor(...colors.darkGray);
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text('Disease Distribution', margin, y);
      y += 10;

      const tableWidth = pageWidth - (margin * 2);
      const colWidths = [tableWidth * 0.5, tableWidth * 0.25, tableWidth * 0.25];
      
      // Table header
      doc.setFillColor(...colors.blue);
      doc.rect(margin, y, tableWidth, 10, 'F');
      
      doc.setTextColor(...colors.white);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('Disease', margin + 5, y + 7);
      doc.text('Cases', margin + colWidths[0] + 5, y + 7);
      doc.text('Percentage', margin + colWidths[0] + colWidths[1] + 5, y + 7);
      
      y += 10;
      
      // Table rows
      const diseaseData = dashboardData?.most_searched_diseases || [];
      const totalDiseaseCases = diseaseData.reduce((sum, d) => sum + (d.count || 0), 0);
      
      diseaseData.forEach((d, idx) => {
        const percentage = totalDiseaseCases > 0 ? ((d.count / totalDiseaseCases) * 100).toFixed(1) : '0.0';
        
        // Alternate row colors
        if (idx % 2 === 0) {
          doc.setFillColor(...colors.lightGray);
          doc.rect(margin, y, tableWidth, 8, 'F');
        }
        
        doc.setTextColor(...colors.darkGray);
        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.text(d.disease || '-', margin + 5, y + 6);
        doc.text(String(d.count || 0), margin + colWidths[0] + 5, y + 6);
        doc.text(`${percentage}%`, margin + colWidths[0] + colWidths[1] + 5, y + 6);
        
        y += 8;
      });
      
      y += 10;

      // ==================== RECENT CASES TABLE ====================
      if (y + 60 > pageHeight) {
        doc.addPage();
        y = margin;
      }
      
      doc.setTextColor(...colors.darkGray);
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text('Recent Cases (Latest 4)', margin, y);
      y += 10;

      // Table header
      doc.setFillColor(...colors.purple);
      doc.rect(margin, y, tableWidth, 10, 'F');
      
      doc.setTextColor(...colors.white);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'bold');
      doc.text('Image Name', margin + 5, y + 7);
      doc.text('Disease', margin + colWidths[0] + 5, y + 7);
      doc.text('Confidence', margin + colWidths[0] + colWidths[1] + 5, y + 7);
      doc.text('Date', margin + colWidths[0] + colWidths[1] + colWidths[2] + 5, y + 7);
      
      y += 10;
      
      // Table rows - latest 4 cases
      const recentCases = predictionHistory.slice(0, 4);
      const caseColWidths = [tableWidth * 0.35, tableWidth * 0.25, tableWidth * 0.15, tableWidth * 0.25];
      
      recentCases.forEach((c, idx) => {
        const date = c.timestamp ? new Date(c.timestamp).toLocaleDateString() : '-';
        const confidence = c.confidence_score ? `${c.confidence_score}%` : '-';
        
        // Alternate row colors
        if (idx % 2 === 0) {
          doc.setFillColor(...colors.lightGray);
          doc.rect(margin, y, tableWidth, 8, 'F');
        }
        
        doc.setTextColor(...colors.darkGray);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        doc.text(c.image_name || '-', margin + 5, y + 6, { maxWidth: caseColWidths[0] - 10 });
        doc.text(c.predicted_disease || '-', margin + caseColWidths[0] + 5, y + 6);
        doc.text(confidence, margin + caseColWidths[0] + caseColWidths[1] + 5, y + 6);
        doc.text(date, margin + caseColWidths[0] + caseColWidths[1] + caseColWidths[2] + 5, y + 6);
        
        y += 8;
      });
      
      y += 15;

      // ==================== AI INSIGHTS BOX ====================
      if (y + 40 > pageHeight) {
        doc.addPage();
        y = margin;
      }
      
      doc.setFillColor(...colors.lightBlue);
      doc.roundedRect(margin, y, tableWidth, 35, 3, 3, 'F');
      
      doc.setFillColor(...colors.blue);
      doc.roundedRect(margin, y, tableWidth, 5, 3, 3, 'F');
      
      doc.setTextColor(...colors.white);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('AI Insights', margin + 5, y + 4);
      
      doc.setTextColor(...colors.darkGray);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      const insights = [
        `Most Common Disease: ${mostCommonDisease}`,
        `Total Predictions: ${totalCases}`,
        `Average Confidence: ${avgConfidence}%`
      ];
      insights.forEach((insight, idx) => {
        doc.text(insight, margin + 5, y + 12 + (idx * 7));
      });
      
      y += 45;

      // ==================== FOOTER ====================
      doc.setFillColor(...colors.lightGray);
      doc.rect(0, pageHeight - 25, pageWidth, 25, 'F');
      
      doc.setTextColor(...colors.darkGray);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.text('Generated by Skin Disease Detection System', pageWidth / 2, pageHeight - 18, { align: 'center' });
      doc.setFontSize(8);
      doc.setTextColor(...colors.blue);
      doc.setFont('helvetica', 'bold');
      doc.text('Confidential Medical Report', pageWidth / 2, pageHeight - 10, { align: 'center' });
      
      // Save PDF
      console.log('Saving PDF...');
      doc.save(`doctor_analytics_report_${reportDate}.pdf`);
      console.log('PDF saved successfully');
    } catch (error) {
      console.error('PDF generation failed:', error);
      console.error('Error details:', error.message, error.stack);
      alert(`Failed to generate PDF: ${error.message}`);
    } finally {
      setDownloadingPdf(false);
    }
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

  const getImageUrlFromCaseItem = (caseItem) => {
    // Prefer backend-provided image_url (already converted to a public URL)
    if (caseItem?.image_url) {
      console.log('[Reports] Using backend image_url:', caseItem.image_url);
      return caseItem.image_url;
    }

    // Fallback: convert stored filesystem path to public URL
    const imageUrlFromPath = getImageUrl(caseItem?.image_path);
    if (imageUrlFromPath) {
      console.log('[Reports] Derived URL from image_path:', imageUrlFromPath);
      return imageUrlFromPath;
    }

    // Last resort: if backend only provides image_name, do NOT build a URL
    // from the original uploaded filename (that file was never saved with
    // that name). Return null so the placeholder is shown instead of a 404.
    console.warn('[Reports] No usable image URL for case:', caseItem);
    return null;
  };

  // Get doctor info from auth/me endpoint
  const [doctorInfo, setDoctorInfo] = useState(null);
  
  useEffect(() => {
    let cancelled = false;
    
    async function loadDoctorInfo() {
      try {
        const response = await api.get('/auth/me');
        const currentUser = response?.data;
        console.log("Current User:", currentUser);
        if (!cancelled) {
          setDoctorInfo(currentUser);
        }
      } catch (e) {
        console.error('Failed to load doctor info:', e);
      }
    }
    
    loadDoctorInfo();
    return () => {
      cancelled = true;
    };
  }, []);

  const reportDate = new Date().toISOString().split('T')[0];

  if (loading) {
    return (
      <Layout title="Reports" userLabel="Doctor" role="doctor">
        <div className="reportsLoading">
          <div className="reportsSpinner" />
          <div className="reportsLoadingText">Loading reports...</div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Reports" userLabel="Doctor" role="doctor">
        <div className="reportsError">
          <div className="reportsErrorTitle">Failed to load reports</div>
          <div className="reportsErrorMessage">{error}</div>
        </div>
      </Layout>
    );
  }

  if (!dashboardData) {
    return (
      <Layout title="Reports" userLabel="Doctor" role="doctor">
        <div className="reportsEmptyState">
          <div className="reportsEmptyText">No data available. Please try again later.</div>
        </div>
      </Layout>
    );
  }
  const predictionHistory = dashboardData?.prediction_history || [];
  const mostSearchedDiseases = dashboardData?.most_searched_diseases || [];
  const totalSearches = dashboardData?.total_searches || 0;
  
  // Display only latest 5 cases
  const displayCases = predictionHistory.length > 5 ? predictionHistory.slice(0, 5) : predictionHistory;
  const showCasesMessage = predictionHistory.length > 5;

  // Calculate AI Insights
  const totalCases = predictionHistory.length;
  const mostCommonDisease = mostSearchedDiseases.length > 0 ? mostSearchedDiseases[0]?.disease : 'N/A';
  const avgConfidence = predictionHistory.length > 0 
    ? (predictionHistory.reduce((sum, c) => sum + (c.confidence_score || 0), 0) / predictionHistory.length).toFixed(2)
    : 0;
  const highestConfidence = predictionHistory.length > 0 
    ? Math.max(...predictionHistory.map(c => c.confidence_score || 0))
    : 0;

  // Prepare chart data for disease distribution
  const chartData = mostSearchedDiseases?.map(item => ({
    name: item.disease,
    value: item.count,
  })) || [];

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={12} fontWeight={600}>
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <Layout title="Reports" userLabel="Doctor" role="doctor">
      <div className="reportsContainer">
        {/* SECTION 1: Premium Report Header */}
        <div className="reportsHeaderCard">
          <div className="reportsHeaderContent">
            <div className="reportsDoctorAvatar">
              <svg className="reportsAvatarIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </div>
            <div className="reportsHeaderInfo">
              <h1 className="reportsHeaderTitle">Doctor Analytics Report</h1>
              <div className="reportsHeaderDetails">
                <div className="reportsHeaderDetail">
                  <span className="reportsHeaderLabel">Doctor:</span>
                  <span className="reportsHeaderValue">{doctorInfo?.full_name || 'N/A'}</span>
                </div>
                <div className="reportsHeaderDetail">
                  <span className="reportsHeaderLabel">Specialization:</span>
                  <span className="reportsHeaderValue">{doctorInfo?.specialization || 'N/A'}</span>
                </div>
                <div className="reportsHeaderDetail">
                  <span className="reportsHeaderLabel">Report Date:</span>
                  <span className="reportsHeaderValue">{reportDate}</span>
                </div>
                <div className="reportsHeaderDetail">
                  <span className="reportsHeaderLabel">Total Cases:</span>
                  <span className="reportsHeaderValue">{totalCases}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* SECTION 2: Analytics Overview Cards */}
        <div className="reportsAnalyticsGrid">
          <div className="reportsAnalyticsCard">
            <div className="reportsAnalyticsIcon reportsAnalyticsIconBlue">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            </div>
            <div className="reportsAnalyticsContent">
              <div className="reportsAnalyticsValue">{totalSearches}</div>
              <div className="reportsAnalyticsLabel">Total Images Analyzed</div>
              <div className="reportsAnalyticsDesc">Images processed by AI</div>
            </div>
          </div>

          <div className="reportsAnalyticsCard">
            <div className="reportsAnalyticsIcon reportsAnalyticsIconGreen">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
            </div>
            <div className="reportsAnalyticsContent">
              <div className="reportsAnalyticsValue">{totalSearches}</div>
              <div className="reportsAnalyticsLabel">Total Reports Generated</div>
              <div className="reportsAnalyticsDesc">Reports created</div>
            </div>
          </div>

          <div className="reportsAnalyticsCard">
            <div className="reportsAnalyticsIcon reportsAnalyticsIconPurple">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            </div>
            <div className="reportsAnalyticsContent">
              <div className="reportsAnalyticsValue">{mostCommonDisease}</div>
              <div className="reportsAnalyticsLabel">Most Detected Disease</div>
              <div className="reportsAnalyticsDesc">Highest frequency</div>
            </div>
          </div>

          <div className="reportsAnalyticsCard">
            <div className="reportsAnalyticsIcon reportsAnalyticsIconOrange">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            </div>
            <div className="reportsAnalyticsContent">
              <div className="reportsAnalyticsValue">
                {avgConfidence}%
              </div>
              <div className="reportsAnalyticsLabel">Average Confidence</div>
              <div className="reportsAnalyticsDesc">AI prediction accuracy</div>
            </div>
          </div>
        </div>

        {/* SECTION 4: Case Analysis Report */}
        <div className="reportsCard">
          <h2 className="reportsCardTitle">Case Analysis Report</h2>
          {displayCases.length === 0 ? (
            <div className="reportsEmptyState">
              <div className="reportsEmptyText">No analyzed cases available yet.</div>
            </div>
          ) : (
            <>
              {showCasesMessage && (
                <div className="reportsInfoMessage">
                  Only the latest 5 cases are displayed in the report.
                </div>
              )}
              <div className="reportsCasesList">
                {displayCases?.map((caseItem, index) => {
                  console.log("Case Item:", caseItem);

                  // Prefer image_path if present; otherwise construct from image_name.
                  const imageUrl = getImageUrlFromCaseItem(caseItem);

                  return (
                    <div key={index} className="reportsCaseCard">
                      <div className="reportsCaseLeft">
                        <div className="reportsCaseNumber">Case #{index + 1}</div>
                        <div className="reportsCaseImageContainer">
                          {imageUrl ? (
                            <img src={imageUrl} alt={caseItem.image_name} className="reportsCaseThumbnail" />
                          ) : (
                            <div className="reportsCaseNoImage">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                                <circle cx="8.5" cy="8.5" r="1.5" />
                                <polyline points="21 15 16 10 5 21" />
                              </svg>
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="reportsCaseRight">
                        <div className="reportsCaseRow">
                          <span className="reportsCaseField">Date & Time:</span>
                          <span className="reportsCaseData">{caseItem.timestamp || 'N/A'}</span>
                        </div>
                        <div className="reportsCaseRow">
                          <span className="reportsCaseField">Image Name:</span>
                          <span className="reportsCaseData">
                            {caseItem.image_name && imageUrl ? (
                              <a
                                href={imageUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{ color: 'inherit', textDecoration: 'underline' }}
                              >
                                {caseItem.image_name}
                              </a>
                            ) : (
                              caseItem.image_name || 'Image Not Available'
                            )}
                          </span>
                        </div>
                        <div className="reportsCaseRow">
                          <span className="reportsCaseField">Predicted Disease:</span>
                          <span className="reportsCaseData reportsCaseDisease">{caseItem.predicted_disease || 'N/A'}</span>
                        </div>
                        <div className="reportsCaseRow">
                          <span className="reportsCaseField">Confidence Score:</span>
                          <span className="reportsCaseData reportsCaseConfidence">{caseItem.confidence_score ? `${caseItem.confidence_score}%` : 'N/A'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* SECTION 3: Disease Distribution */}
        <div className="reportsCard reportsCardLarge">
          <h2 className="reportsCardTitle">Disease Distribution</h2>
          {mostSearchedDiseases.length === 0 ? (
            <div className="reportsEmptyState">
              <div className="reportsEmptyText">No disease distribution available.</div>
            </div>
          ) : (
            <div className="reportsDistributionLayout">
              <div className="reportsDistributionChart">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={renderCustomizedLabel}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData?.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="reportsDistributionTable">
                <div className="reportsDistributionHeader">
                  <div className="reportsDistributionCell reportsDistributionHeaderCell">Disease Name</div>
                  <div className="reportsDistributionCell reportsDistributionHeaderCell">Cases</div>
                  <div className="reportsDistributionCell reportsDistributionHeaderCell">Percentage</div>
                </div>
                {mostSearchedDiseases?.map((item, index) => {
                  const percentage = totalCases > 0 ? ((item.count / totalCases) * 100).toFixed(1) : 0;
                  return (
                    <div key={index} className="reportsDistributionRow">
                      <div className="reportsDistributionCell">
                        <div className="reportsDistributionColor" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                        {item.disease}
                      </div>
                      <div className="reportsDistributionCell">{item.count}</div>
                      <div className="reportsDistributionCell">{percentage}%</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* SECTION 5: AI Insights Panel */}
        <div className="reportsInsightsCard">
          <div className="reportsInsightsHeader">
            <div className="reportsInsightsIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" />
                <path d="M12 6v6l4 2" />
              </svg>
            </div>
            <h3 className="reportsInsightsTitle">AI Insights</h3>
          </div>
          <div className="reportsInsightsGrid">
            <div className="reportsInsightItem">
              <div className="reportsInsightLabel">Most Common Disease</div>
              <div className="reportsInsightValue">{mostCommonDisease}</div>
            </div>
            <div className="reportsInsightItem">
              <div className="reportsInsightLabel">Highest Confidence</div>
              <div className="reportsInsightValue">{highestConfidence}%</div>
            </div>
            <div className="reportsInsightItem">
              <div className="reportsInsightLabel">Total Analyzed</div>
              <div className="reportsInsightValue">{totalCases}</div>
            </div>
            <div className="reportsInsightItem">
              <div className="reportsInsightLabel">Average Confidence</div>
              <div className="reportsInsightValue">{avgConfidence}%</div>
            </div>
          </div>
        </div>

        {/* SECTION 6: PDF Download Action */}
        <div className="reportsCard">
          <button
            className="reportsDownloadButton"
            onClick={handleDownloadPdf}
            disabled={downloadingPdf}
          >
            <svg className="reportsDownloadIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 12 12 17 17 12" />
              <line x1="12" y1="3" x2="12" y2="17" />
            </svg>
            {downloadingPdf ? 'Generating PDF...' : 'PDF Download'}
          </button>
        </div>
      </div>
    </Layout>
  );
}
