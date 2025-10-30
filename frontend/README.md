# Reviewer Recommendation Frontend

A minimal, clean React application for the Academic Reviewer Recommendation System.

## Features

- **Dual Input Modes:**
  - 📎 Upload PDF files
  - ✏️ Paste title and abstract directly

- **Smart Recommendations:**
  - LightGBM-powered ranking
  - Conflict-of-interest detection
  - Evidence-based suggestions

- **Rich UI:**
  - Collapsible evidence lists
  - Score visualizations (no external chart library)
  - Responsive design
  - Error handling

## Setup

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Or with yarn
yarn install
```

### Configuration

Create a `.env` file in the root directory:

```env
VITE_API_BASE=http://localhost:8000
```

## Development

```bash
# Start development server
npm run dev

# Or with yarn
yarn dev
```

The app will be available at `http://localhost:5173`

## Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── FileUploader.jsx      # PDF upload component
│   │   ├── FileUploader.css
│   │   ├── PasteAbstract.jsx     # JSON input component
│   │   ├── PasteAbstract.css
│   │   ├── ReviewerList.jsx      # Reviewer table with evidence
│   │   ├── ReviewerList.css
│   │   ├── ScoreChart.jsx        # Score visualization (div-based)
│   │   └── ScoreChart.css
│   ├── App.jsx                   # Main app with tabs
│   ├── App.css
│   ├── main.jsx                  # Entry point
│   └── index.css                 # Global styles
├── public/
├── index.html
├── package.json
├── vite.config.js
└── .env
```

## Components

### FileUploader

Handles PDF file uploads with drag-and-drop support.

**Props:**
- `apiBase`: API base URL
- `onRecommendations`: Callback for successful recommendations
- `onError`: Callback for errors
- `onLoadingChange`: Callback for loading state changes

### PasteAbstract

Handles JSON input for title and abstract.

**Props:**
- Same as FileUploader

### ReviewerList

Displays recommendations in a table with:
- Rank badges
- Name and affiliation
- Score values
- Score visualization bars
- Collapsible evidence lists

**Props:**
- `reviewers`: Array of reviewer objects

### ScoreChart

Displays a horizontal bar chart using pure CSS (no libraries).

**Props:**
- `score`: Current reviewer score
- `maxScore`: Maximum score for normalization

## API Integration

The app uses Axios to communicate with the backend API:

### Endpoints Used

**POST /recommend** (File Upload)
```javascript
const formData = new FormData()
formData.append('file', pdfFile)
formData.append('authors', 'John Smith, Jane Doe')
formData.append('k', 10)

axios.post(`${API_BASE}/recommend`, formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
})
```

**POST /recommend** (JSON)
```javascript
axios.post(`${API_BASE}/recommend`, {
  title: 'Paper Title',
  abstract: 'Paper abstract...',
  authors: ['John Smith', 'Jane Doe'],
  k: 10
}, {
  headers: { 'Content-Type': 'application/json' }
})
```

## Error Handling

The app handles various error scenarios:

- ✅ Invalid file types
- ✅ Missing required fields
- ✅ API connection errors
- ✅ Backend errors (shows error.response.data.detail)
- ✅ Network failures

All errors are displayed in a user-friendly error box.

## Styling

- Pure CSS (no CSS frameworks)
- Responsive design (mobile-friendly)
- Gradient backgrounds
- Smooth animations
- Accessible color contrasts

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

MIT
