import { useState } from 'react'
import FileUploader from './components/FileUploader'
import ReviewerList from './components/ReviewerList'
import EvaluationReport from './components/EvaluationReport'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function App() {
  const [recommendations, setRecommendations] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleRecommendationsReceived = (data) => {
    setRecommendations(data)
    setError(null)
  }

  const handleError = (err) => {
    setError(err)
    setRecommendations(null)
  }

  const handleReset = () => {
    setRecommendations(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>📄 Reviewer Recommendation System</h1>
        <p>Find the best reviewers for your academic paper</p>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <FileUploader
            apiBase={API_BASE}
            onRecommendations={handleRecommendationsReceived}
            onError={handleError}
            onLoadingChange={setLoading}
          />
        </div>

        {/* Evaluation Report */}
        <EvaluationReport apiBase={API_BASE} />

        {error && (
          <div className="error-box">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="loading-box">
            <div className="spinner"></div>
            <p>Finding the best reviewers for your paper...</p>
          </div>
        )}

        {recommendations && !loading && (
          <div className="results-section">
            <div className="results-header">
              <h2>🎯 Recommended Reviewers</h2>
              <div className="results-stats">
                <span className="stat">
                  <strong>{recommendations.recommendations.length}</strong> recommendations
                </span>
                <span className="stat">
                  <strong>{recommendations.total_candidates}</strong> candidates analyzed
                </span>
                <span className="stat">
                  <strong>{recommendations.filtered_by_coi}</strong> filtered by COI
                </span>
                <span className="stat-model">
                  Model: <strong>{recommendations.model_used}</strong>
                </span>
              </div>
            </div>
            
            <ReviewerList reviewers={recommendations.recommendations} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by LightGBM, SciBERT, and FAISS | Academic Reviewer Recommendation System</p>
      </footer>
    </div>
  )
}

export default App
