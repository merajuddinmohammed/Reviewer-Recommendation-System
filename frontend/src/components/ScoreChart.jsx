import './ScoreChart.css'

function ScoreChart({ score, maxScore }) {
  // Calculate percentage relative to max score
  const percentage = maxScore > 0 ? (score / maxScore) * 100 : 0
  
  // Determine color based on score percentage
  const getColor = (pct) => {
    if (pct >= 80) return '#4caf50' // Green
    if (pct >= 60) return '#8bc34a' // Light green
    if (pct >= 40) return '#ffc107' // Yellow
    if (pct >= 20) return '#ff9800' // Orange
    return '#f44336' // Red
  }

  const color = getColor(percentage)

  return (
    <div className="score-chart">
      <div className="score-bar-container">
        <div 
          className="score-bar-fill"
          style={{ 
            width: `${percentage}%`,
            backgroundColor: color
          }}
        >
          <span className="score-bar-label">{percentage.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  )
}

export default ScoreChart
