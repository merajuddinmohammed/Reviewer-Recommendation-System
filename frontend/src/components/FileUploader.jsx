import { useState, useRef } from 'react'
import axios from 'axios'
import './FileUploader.css'

function FileUploader({ apiBase, onRecommendations, onError, onLoadingChange }) {
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [authors, setAuthors] = useState('')
  const [affiliations, setAffiliations] = useState('')
  const [numRecommendations, setNumRecommendations] = useState(10)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile)
      } else {
        onError('Please upload a PDF file')
      }
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile)
        onError(null)
      } else {
        onError('Please upload a PDF file')
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!file) {
      onError('Please select a PDF file')
      return
    }

    onLoadingChange(true)
    onError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('k', numRecommendations)
      
      if (authors.trim()) {
        formData.append('authors', authors.trim())
      }
      
      if (affiliations.trim()) {
        formData.append('affiliations', affiliations.trim())
      }

      const response = await axios.post(`${apiBase}/recommend`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
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
    setFile(null)
    setAuthors('')
    setAffiliations('')
    setNumRecommendations(10)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="file-uploader">
      <form onSubmit={handleSubmit}>
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          
          {file ? (
            <div className="file-info">
              <div className="file-icon">📄</div>
              <div className="file-details">
                <p className="file-name">{file.name}</p>
                <p className="file-size">{(file.size / 1024).toFixed(2)} KB</p>
              </div>
              <button 
                type="button" 
                className="remove-file"
                onClick={(e) => {
                  e.stopPropagation()
                  handleReset()
                }}
              >
                ✕
              </button>
            </div>
          ) : (
            <>
              <div className="upload-icon">📎</div>
              <p className="upload-text">
                <strong>Click to upload</strong> or drag and drop
              </p>
              <p className="upload-hint">PDF files only</p>
            </>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="authors">
            <span className="label-icon">👤</span>
            Authors (optional)
          </label>
          <input
            id="authors"
            type="text"
            value={authors}
            onChange={(e) => setAuthors(e.target.value)}
            placeholder="e.g., John Smith, Jane Doe"
            className="form-input"
          />
          <small>Comma-separated list for conflict-of-interest detection</small>
        </div>

        <div className="form-group">
          <label htmlFor="affiliations">
            <span className="label-icon">🏛️</span>
            Affiliations (optional)
          </label>
          <input
            id="affiliations"
            type="text"
            value={affiliations}
            onChange={(e) => setAffiliations(e.target.value)}
            placeholder="e.g., MIT, Stanford University"
            className="form-input"
          />
          <small>Comma-separated list for conflict-of-interest detection</small>
        </div>

        <div className="form-group">
          <label htmlFor="numRecs">
            <span className="label-icon">🔢</span>
            Number of Recommendations
          </label>
          <input
            id="numRecs"
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
          <button type="submit" className="btn btn-primary" disabled={!file}>
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

export default FileUploader
