## Plan: Fix Admin Dashboard summary cards clipping at 100% zoom

### Information gathered
- Summary cards for the admin “System Monitoring” page are implemented in `frontend/src/pages/admin/Dashboard.jsx` and styled in `frontend/src/pages/admin/Dashboard.css`.
- Existing card CSS (`.adminAnalyticsCard`) uses `display:flex` with `align-items:center` and fixed-ish typography sizes (e.g. `.adminAnalyticsValue { font-size: 28px; }`, `.adminAnalyticsLabel { font-size: 13px; }`).
- The same file does **not** show any explicit truncation CSS (`text-overflow: ellipsis`, `overflow: hidden`, `white-space: nowrap`).
- The reusable `StatCard` component (`frontend/src/components/dashboard/StatCard.css`) also contains large fixed font sizing but similarly shows no explicit truncation.

### Plan
1. Update `frontend/src/pages/admin/Dashboard.css` summary card styles so long disease names / values never get clipped at 100% zoom:
   - Make card content more flexible: change `.adminAnalyticsCard` alignment to `align-items:flex-start`.
   - Ensure `.adminAnalyticsContent` can grow and wraps: set `min-width:0`, and apply wrapping behavior on value/label elements.
   - Remove any implicit single-line constraints by explicitly setting:
     - `white-space: normal;`
     - `overflow-wrap: anywhere;`
     - `word-break: break-word;`
   - Avoid fixed height issues while keeping equal height across cards:
     - Ensure cards stretch to the same grid row height via `align-items:stretch` on `.adminDashboardCards`.
     - Add a consistent `min-height`/`height:100%` strategy so all cards are equal height.
   - Make typography responsive using `clamp()` for `.adminAnalyticsValue`, `.adminAnalyticsLabel`, and `.adminAnalyticsDesc`.
2. Keep spacing equal between cards by preserving the grid `gap` and ensuring card internal layout uses consistent padding/gaps.
3. Confirm no backend/layout/chart/sidebar/routing changes are made.
4. Smoke test: run frontend dev/build and verify at 100% browser zoom that long labels wrap and are fully visible.

### Dependent files to edit
- `frontend/src/pages/admin/Dashboard.css`

### Followup steps after editing
- Run `npm` scripts for the frontend (build or dev) to ensure no CSS syntax errors.

<ask_followup_question>
Proceed with edits to `frontend/src/pages/admin/Dashboard.css` only (summary cards CSS) to fix wrapping/clipping at 100% zoom, without changing JSX structure or card order?
</ask_followup_question>

