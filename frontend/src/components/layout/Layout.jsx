import Header from './Header';
import Sidebar from './Sidebar';

import './Layout.css';

export default function Layout({
  children,
  title = 'Dashboard',
  userLabel = 'Admin',
  role = 'admin',
}) {
  return (
    <div className="appLayout">
      <Sidebar role={role} />

      <div className="appMain">
        <Header
          title={title}
          userLabel={userLabel}
        />

        <main className="appContent">
          {children}
        </main>
      </div>
    </div>
  );
}

