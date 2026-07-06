import { useEffect, useState } from 'react';
import Layout from '../../components/layout/Layout';
import api from '../../services/api';
import './ManageDoctors.css';

export default function ManageDoctors() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [doctors, setDoctors] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    let cancelled = false;

    async function loadDoctors() {
      try {
        const response = await api.get('/admin/doctors');
        const data = response?.data || [];
        console.log("Doctors Response:", data);
        if (!cancelled) {
          setDoctors(data);
        }
      } catch (e) {
        console.error('Failed to load doctors:', e);
        if (!cancelled) {
          setError('Failed to load doctors data');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadDoctors();
    return () => {
      cancelled = true;
    };
  }, []);

  // Filter doctors based on search term
  const filteredDoctors = doctors.filter(doctor => {
    const searchLower = searchTerm.toLowerCase();
    return (
      doctor.doctor_name?.toLowerCase().includes(searchLower) ||
      doctor.email?.toLowerCase().includes(searchLower)
    );
  });

  if (loading) {
    return (
      <Layout title="Doctor Management" userLabel="Admin">
        <div className="manageDoctors">
          <div className="manageDoctorsHeader">
            <div className="manageDoctorsTitleWrap">
              <div className="manageDoctorsTitle">Doctor Management</div>
              <div className="manageDoctorsSubtitle">Loading...</div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Doctor Management" userLabel="Admin">
        <div className="manageDoctors">
          <div className="manageDoctorsHeader">
            <div className="manageDoctorsTitleWrap">
              <div className="manageDoctorsTitle">Doctor Management</div>
              <div className="manageDoctorsSubtitle">Error</div>
            </div>
          </div>
          <div className="manageDoctorsError">{error}</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Doctor Management" userLabel="Admin">
      <div className="manageDoctors">
        <header className="manageDoctorsHeader">
          <div className="manageDoctorsTitleWrap">
            <div className="manageDoctorsTitle">Doctor Management</div>
            <div className="manageDoctorsSubtitle">Manage registered doctors</div>
          </div>
        </header>

        {/* Summary Card */}
        <section className="manageDoctorsSummary">
          <div className="manageDoctorsSummaryCard">
            <div className="manageDoctorsSummaryIcon">👨‍⚕️</div>
            <div className="manageDoctorsSummaryContent">
              <div className="manageDoctorsSummaryValue">{doctors.length}</div>
              <div className="manageDoctorsSummaryLabel">Total Registered Doctors</div>
            </div>
          </div>
        </section>

        {/* Search */}
        <section className="manageDoctorsSearch">
          <input
            type="text"
            placeholder="Search by doctor name or email..."
            className="manageDoctorsSearchInput"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </section>

        {/* Doctors Table */}
        <section className="manageDoctorsTableSection">
          <div className="manageDoctorsSectionHeader">
            <div className="manageDoctorsSectionTitle">Registered Doctors</div>
          </div>

          <div className="manageDoctorsTableWrap">
            <table className="manageDoctorsTable">
              <thead>
                <tr>
                  <th>Doctor Name</th>
                  <th>Email</th>
                  <th>Total Images Analyzed</th>
                </tr>
              </thead>
              <tbody>
                {filteredDoctors.length > 0 ? (
                  filteredDoctors.map((doctor) => (
                    <tr key={doctor.id}>
                      <td style={{ color: '#1F2937', fontWeight: 600 }}>{doctor.doctor_name || '-'}</td>
                      <td style={{ color: '#374151' }}>{doctor.email || '-'}</td>
                      <td style={{ color: '#111827', fontWeight: 700 }}>{doctor.total_images_analyzed || 0}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3" style={{ textAlign: 'center', padding: '32px', color: '#6B7280' }}>
                      No doctors found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </Layout>
  );
}
