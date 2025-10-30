import { useState } from 'react'
import axios from 'axios'
import './PasteAbstract.css'

function PasteAbstract({ apiBase, onRecommendations, onError, onLoadingChange }) {
  const [title, setTitle] = useState('')
  const [abstract, setAbstract] = useState('')
  const [authors, setAuthors] = useState('')
  const [affiliations, setAffiliations] = useState('')
  const [numRecommendations, setNumRecommendations] = useState(10)

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!title.trim() || !abstract.trim()) {
      onError('Please provide both title and abstract')
      return
    }

    onLoadingChange(true)
    onError(null)

    try {
      // Parse authors and affiliations
      const authorsList = authors.trim() 
        ? authors.split(',').map(a => a.trim()).filter(a => a)
        : []
      
      const affiliationsList = affiliations.trim()
        ? affiliations.split(',').map(a => a.trim()).filter(a => a)
        : []

      const response = await axios.post(`${apiBase}/recommend`, {
        title: title.trim(),
        abstract: abstract.trim(),
        authors: authorsList.length > 0 ? authorsList : undefined,
        affiliations: affiliationsList.length > 0 ? affiliationsList : undefined,
        k: numRecommendations
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      })

      onRecommendations(response.data)
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to get recommendations'
      onError(errorMessage)
    } finally {
      onLoadingChange(false)
    }
  }

  const handleReset = () => {
    setTitle('')
    setAbstract('')
    setAuthors('')
    setAffiliations('')
    setNumRecommendations(10)
  }

  return (
    <div className="paste-abstract">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="title">
            <span className="label-icon">📝</span>
            Paper Title *
          </label>
          <input
            id="title"
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter your paper title"
            className="form-input"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="abstract">
            <span className="label-icon">📄</span>
            Abstract *
          </label>
          <textarea
            id="abstract"
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
            placeholder="Paste your paper abstract here..."
            className="form-textarea"
            rows={8}
            required
          />
          <small>{abstract.length} characters</small>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="authors-paste">
              <span className="label-icon">👤</span>
              Authors (optional)
            </label>
            <input
              id="authors-paste"
              type="text"
              value={authors}
              onChange={(e) => setAuthors(e.target.value)}
              placeholder="e.g., John Smith, Jane Doe"
              className="form-input"
            />
            <small>Comma-separated for COI detection</small>
          </div>

          <div className="form-group">
            <label htmlFor="affiliations-paste">
              <span className="label-icon">🏛️</span>
              Affiliations (optional)
            </label>
            <input
              id="affiliations-paste"
              type="text"
              value={affiliations}
              onChange={(e) => setAffiliations(e.target.value)}
              placeholder="e.g., MIT, Stanford"
              className="form-input"
            />
            <small>Comma-separated for COI detection</small>
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="numRecs-paste">
            <span className="label-icon">🔢</span>
            Number of Recommendations
          </label>
          <input
            id="numRecs-paste"
            type="number"
            min="1"
            max="50"
            value={numRecommendations}
            onChange={(e) => setNumRecommendations(parseInt(e.target.value))}
            className="form-input"
          />
          <small>Between 1 and 50 recommendations</small>
        </div>

        <div className="button-group">
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={!title.trim() || !abstract.trim()}
          >
            🎯 Get Recommendations
          </button>
          <button type="button" className="btn btn-secondary" onClick={handleReset}>
            🔄 Reset
          </button>
        </div>
      </form>
    </div>
  )
}

export default PasteAbstract
