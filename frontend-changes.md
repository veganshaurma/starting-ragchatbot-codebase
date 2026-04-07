# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a dark/light theme toggle button that lets users switch between the existing dark theme and a new light theme. The selected theme is persisted via `localStorage`.

---

## Files Modified

### `frontend/style.css`
- **Light theme variables** — Added `[data-theme="light"]` block with a full set of overriding CSS custom properties:
  - `--background: #f8fafc`, `--surface: #ffffff`, `--surface-hover: #f1f5f9`
  - `--text-primary: #0f172a`, `--text-secondary: #64748b`
  - `--border-color: #e2e8f0`
  - `--shadow` reduced opacity for light contexts
  - `--welcome-bg: #eff6ff`, `--welcome-border: #93c5fd`
- **Global theme transitions** — Added `body, body *` rule with `transition` on `background-color`, `color`, `border-color`, and `box-shadow` (0.3s ease) so switching is smooth.
- **`.theme-toggle` button styles** — Fixed-position button (top-right, `z-index: 100`), circular, 40×40 px, using surface/border variables. Hover lifts with scale + primary-color border. Focus ring uses `--focus-ring`.
- **Icon animation** — Sun and moon SVGs are absolutely stacked; CSS toggles `opacity` and `rotate/scale` transforms based on `[data-theme="light"]` presence, creating a smooth swap animation.
- **Code block overrides** — Added `[data-theme="light"]` rules to lighten the semi-transparent `rgba(0,0,0,...)` backgrounds on `code` and `pre` elements for better contrast in light mode.

### `frontend/index.html`
- **Anti-flash inline script** — Added a `<script>` tag before the stylesheet that reads `localStorage.getItem('theme')` and immediately sets `data-theme` on `<html>`, preventing a dark→light flash on page load.
- **Toggle button markup** — Added `.theme-toggle#themeToggle` button before `.container`, containing two stacked SVG icons: a sun (`.icon-sun`) and a moon (`.icon-moon`). Includes `aria-label` and `title` for accessibility.
- Bumped stylesheet cache-buster from `v=10` to `v=11`.

### `frontend/script.js`
- **`themeToggle` DOM ref** — Added to the element cache at the top.
- **Saved theme on init** — In `DOMContentLoaded`, reads `localStorage.getItem('theme')` and applies it via `document.documentElement.setAttribute('data-theme', ...)` (redundant safety net alongside the inline script).
- **Toggle handler** — Attached to `#themeToggle` click: reads current `data-theme`, toggles between `'light'` and absent (dark), and syncs to `localStorage`.
