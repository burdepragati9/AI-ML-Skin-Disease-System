import './Header.css';

export default function Header({ title = 'Dashboard', userLabel = 'Admin' }) {
  return (
    <header className="header" aria-label="Top header">
      <div className="headerInner">
        <div className="headerTitle" aria-label="Current page title">
          {title}
        </div>
        <div className="headerUser" aria-label="Logged in user">
          {userLabel}
        </div>
      </div>
    </header>
  );
}

