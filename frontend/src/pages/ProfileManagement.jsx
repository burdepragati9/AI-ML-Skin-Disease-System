import { useEffect, useState } from 'react';
import Layout from '../components/layout/Layout';
import api from '../services/api';
import './ProfileManagement.css';

export default function ProfileManagement() {
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [profile, setProfile] = useState(null);

  const [formData, setFormData] = useState({
    full_name: '',
    specialization: '',
    clinic_name: '',
    phone: '',
    experience: '',
    location: '',
  });

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      setLoading(true);
      setError('');
      try {
        const response = await api.get('/profile/me');
        const data = response.data;

        if (!cancelled) {
          setProfile(data);
          setFormData({
            full_name: data.full_name || '',
            specialization: data.specialization || '',
            clinic_name: data.clinic_name || '',
            phone: data.phone || '',
            experience: data.experience || '',
            location: data.location || '',
          });
        }
      } catch (e) {
        if (!cancelled) {
          const msg = e?.response?.data?.detail || e?.message || 'Failed to load profile';
          setError(msg);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadProfile();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setUpdating(true);
    setError('');
    setSuccess('');

    try {
      const updateData = {
        full_name: formData.full_name,
        specialization: formData.specialization,
        clinic_name: formData.clinic_name,
        phone: formData.phone,
        experience: formData.experience ? parseInt(formData.experience, 10) : 0,
        location: formData.location,
      };

      const response = await api.put('/profile/update', updateData);
      setProfile(response.data);
      setSuccess('Profile updated successfully!');
      
      setTimeout(() => setSuccess(''), 3000);
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || 'Failed to update profile';
      setError(msg);
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <Layout title="Profile Management" userLabel="Doctor" role="doctor">
        <div className="pmCenter">
          <div className="pmSpinner" />
          <div className="pmMuted">Loading profile...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Profile Management" userLabel="Doctor" role="doctor">
      <div className="pmPage">
        <header className="pmHeader">
          <div className="pmTitleWrap">
            <div className="pmTitle">Profile Management</div>
            <div className="pmSubtitle">View and update your doctor profile</div>
          </div>
        </header>

        {error && (
          <div className="pmErrorWrap">
            <div className="pmErrorTitle">Error</div>
            <div className="pmErrorMsg">{error}</div>
          </div>
        )}

        {success && (
          <div className="pmSuccessWrap">
            <div className="pmSuccessTitle">Success</div>
            <div className="pmSuccessMsg">{success}</div>
          </div>
        )}

        <div className="pmGrid">
          {/* Section 1: Doctor Profile (Read-only) */}
          <section className="pmSection glassCard">
            <div className="pmSectionHeader">
              <div className="pmSectionTitle">Doctor Profile</div>
              <div className="pmSectionHint">View your current information</div>
            </div>

            {profile && (
              <div className="pmProfileInfo">
                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Full Name</div>
                  <div className="pmInfoValue">{profile.full_name || 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Email</div>
                  <div className="pmInfoValue">{profile.email || 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Specialization</div>
                  <div className="pmInfoValue">{profile.specialization || 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Doctor ID</div>
                  <div className="pmInfoValue">{profile.doctor_id || 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Hospital/Clinic</div>
                  <div className="pmInfoValue">{profile.clinic_name || 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Phone Number</div>
                  <div className="pmInfoValue">{profile.phone || 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Experience</div>
                  <div className="pmInfoValue">{profile.experience ? `${profile.experience} years` : 'N/A'}</div>
                </div>

                <div className="pmInfoRow">
                  <div className="pmInfoLabel">Location</div>
                  <div className="pmInfoValue">{profile.location || 'N/A'}</div>
                </div>
              </div>
            )}
          </section>

          {/* Section 2: Edit Profile Form */}
          <section className="pmSection glassCard">
            <div className="pmSectionHeader">
              <div className="pmSectionTitle">Edit Profile</div>
              <div className="pmSectionHint">Update your information</div>
            </div>

            <form className="pmForm" onSubmit={handleSubmit}>
              <div className="pmFormGroup">
                <label className="pmFormLabel">Full Name</label>
                <input
                  type="text"
                  name="full_name"
                  className="pmFormInput"
                  value={formData.full_name}
                  onChange={handleChange}
                  placeholder="Enter your full name"
                />
              </div>

              <div className="pmFormGroup">
                <label className="pmFormLabel">Specialization</label>
                <input
                  type="text"
                  name="specialization"
                  className="pmFormInput"
                  value={formData.specialization}
                  onChange={handleChange}
                  placeholder="Enter your specialization"
                />
              </div>

              <div className="pmFormGroup">
                <label className="pmFormLabel">Hospital Name</label>
                <input
                  type="text"
                  name="clinic_name"
                  className="pmFormInput"
                  value={formData.clinic_name}
                  onChange={handleChange}
                  placeholder="Enter hospital/clinic name"
                />
              </div>

              <div className="pmFormGroup">
                <label className="pmFormLabel">Phone Number</label>
                <input
                  type="text"
                  name="phone"
                  className="pmFormInput"
                  value={formData.phone}
                  onChange={handleChange}
                  placeholder="Enter phone number"
                />
              </div>

              <div className="pmFormGroup">
                <label className="pmFormLabel">Experience (years)</label>
                <input
                  type="number"
                  name="experience"
                  className="pmFormInput"
                  value={formData.experience}
                  onChange={handleChange}
                  placeholder="Enter years of experience"
                  min="0"
                />
              </div>

              <div className="pmFormGroup">
                <label className="pmFormLabel">Location</label>
                <input
                  type="text"
                  name="location"
                  className="pmFormInput"
                  value={formData.location}
                  onChange={handleChange}
                  placeholder="Enter your location"
                />
              </div>

              <button
                type="submit"
                className="pmBtn pmBtnPrimary"
                disabled={updating}
              >
                {updating ? 'Updating...' : 'Update Profile'}
              </button>
            </form>
          </section>
        </div>
      </div>
    </Layout>
  );
}
