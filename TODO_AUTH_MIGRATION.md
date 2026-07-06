# Auth migration - checklist

- [x] Create `backend/routes/auth.py` with `POST /auth/login`
- [x] Create `backend/services/doctor_auth.py`
- [x] Create `backend/services/admin_auth.py`
- [x] Update `backend/main.py` to include auth router (`/auth`)
- [x] Update `frontend/src/pages/Login.jsx` to call `POST /auth/login` and navigate to `/dashboard`

