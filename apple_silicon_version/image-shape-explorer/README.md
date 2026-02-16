# Image Shape Explorer (React + Vite)

Frontend UI for visualizing Crofton boundary detection results from the Metal-accelerated Flask backend.

## Requirements

- Node.js 18+
- npm 9+

## Local development

From the repository root:

```sh
cd apple_silicon_version/image-shape-explorer
npm install
npm run dev
```

## Backend URL configuration

By default, the worker targets `http://localhost:65060` and falls back to port `5000`.

To override this, create `apple_silicon_version/image-shape-explorer/.env.local`:

```sh
VITE_CROFTON_BACKEND_URL=http://localhost:65060
```

## Build

```sh
npm run build
npm run preview
```
