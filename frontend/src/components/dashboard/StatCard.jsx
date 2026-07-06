import PropTypes from 'prop-types';

import './StatCard.css';

export default function StatCard({ icon, label, value, tone = 'blue' }) {
  return (
    <section className={`statCard statCard--${tone}`} aria-label={label}>
      <div className="statCardTop">
        <div className="statCardIcon" aria-hidden="true">
          {icon}
        </div>
        <div className="statCardLabel">{label}</div>
      </div>
      <div className="statCardValue">{value}</div>
    </section>
  );
}

StatCard.propTypes = {
  icon: PropTypes.node.isRequired,
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  tone: PropTypes.oneOf(['blue', 'green', 'purple', 'teal', 'rose']),
};

