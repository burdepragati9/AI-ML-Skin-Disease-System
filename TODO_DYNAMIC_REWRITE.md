# TODO - Signup form rendering fix (submit button)

- [ ] Step 0: Add debug logs and ensure signup form uses `with st.form("signup_form"):` and includes `st.form_submit_button("Create Account")`.
- [ ] Step 1: Ensure ALL signup fields are inside the same `with st.form(...)` block (including Full Name, Email, Username, Specialization, Password, Confirm Password, optional Profile Photo Upload).
- [ ] Step 2: Remove the fallback error message “Signup form failed to load” once rendering is deterministic.
- [ ] Step 3: Add debug logs for form rendering, submit click, and signup validation.
- [ ] Step 4: Keep backend/authentication logic unchanged; only refactor UI/Streamlit form structure.

