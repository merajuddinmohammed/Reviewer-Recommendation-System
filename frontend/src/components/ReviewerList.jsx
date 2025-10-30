import { useState } from 'react'
import ScoreChart from './ScoreChart'
import './ReviewerList.css'

function ReviewerList({ reviewers }) {
  const [expandedReviewer, setExpandedReviewer] = useState(null)

  const toggleEvidence = (authorId) => {
    setExpandedReviewer(expandedReviewer === authorId ? null : authorId)
  }

  if (!reviewers || reviewers.length === 0) {
    return (
      <div className="no-reviewers">
        <p>No reviewers found</p>
      </div>
    )
  }

  return (
    <div className="reviewer-list">
      <table className="reviewer-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>Affiliation</th>
            <th>Score</th>
            <th>Score Visualization</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {reviewers.map((reviewer, index) => (
            <tr key={reviewer.author_id} className="reviewer-row">
              <td className="rank-cell">
                <div className="rank-badge">#{index + 1}</div>
              </td>
              <td className="name-cell">
                <div className="reviewer-name">{reviewer.name}</div>
              </td>
              <td className="affiliation-cell">
                {reviewer.affiliation || (
                  <span className="no-affiliation">Not specified</span>
                )}
              </td>
              <td className="score-cell">
                <span className="score-value">
                  {reviewer.score.toFixed(3)}
                </span>
              </td>
              <td className="chart-cell">
                <ScoreChart score={reviewer.score} maxScore={reviewers[0].score} />
              </td>
              <td className="evidence-cell">
                <button
                  className={`evidence-toggle ${expandedReviewer === reviewer.author_id ? 'expanded' : ''}`}
                  onClick={() => toggleEvidence(reviewer.author_id)}
                >
                  {reviewer.evidence.length > 0 ? (
                    <>
                      {expandedReviewer === reviewer.author_id ? '▼' : '▶'} 
                      {' '}{reviewer.evidence.length} paper{reviewer.evidence.length !== 1 ? 's' : ''}
                    </>
                  ) : (
                    'No evidence'
                  )}
                </button>
                
                {expandedReviewer === reviewer.author_id && reviewer.evidence.length > 0 && (
                  <div className="evidence-list">
                    <h4>Top Matching Papers:</h4>
                    <ul>
                      {reviewer.evidence.map((paper, idx) => (
                        <li key={idx} className="evidence-item">
                          <div className="evidence-header">
                            <span className="evidence-title">{paper.paper_title}</span>
                            <span className="evidence-similarity">
                              {(paper.similarity * 100).toFixed(1)}% match
                            </span>
                          </div>
                          <div className="evidence-meta">
                            {paper.year && (
                              <span className="evidence-year">📅 {paper.year}</span>
                            )}
                          </div>
                          <div className="similarity-bar">
                            <div 
                              className="similarity-fill"
                              style={{ width: `${paper.similarity * 100}%` }}
                            />
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default ReviewerList
