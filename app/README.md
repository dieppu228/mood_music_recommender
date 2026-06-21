# VibeCue React UI

React and Vite frontend for the local Music Mood Agent API.

## Development

Start the API on port 8000, then run:

```bash
cd app
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/v1` and `/health` to the API.

## Production build

```bash
cd app
npm run build
```

FastAPI serves the generated `app/dist` bundle at `http://localhost:8000`.

## VS Code Live Server

Build the UI, then use **Open with Live Server** on `app/dist/index.html`.
The Live Server build calls the API at `http://127.0.0.1:8000`.
