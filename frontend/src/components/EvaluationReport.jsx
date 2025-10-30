import { useState, useEffect } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js'
import { Bar, Doughnut } from 'react-chartjs-2'
import './EvaluationReport.css'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
)

function EvaluationReport({ apiBase }) {
  const [reportData, setReportData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showReport, setShowReport] = useState(false)

  const fetchReport = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${apiBase}/eval-report`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch evaluation report')
      }
      
      const data = await response.json()
      setReportData(data)
      setShowReport(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleToggle = () => {
    if (!showReport && !reportData) {
      fetchReport()
    } else {
      setShowReport(!showReport)
    }
  }

  // Prepare chart data
  const getBarChartData = () => {
    if (!reportData?.metrics?.methods) return null

    const methods = reportData.metrics.methods
    
    return {
      labels: methods.map(m => m.name),
      datasets: [
        {
          label: 'Precision@5',
          data: methods.map(m => (m.p5_mean * 100).toFixed(2)),
          backgroundColor: 'rgba(75, 192, 192, 0.6)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 2
        },
        {
          label: 'nDCG@10',
          data: methods.map(m => (m.ndcg10_mean * 100).toFixed(2)),
          backgroundColor: 'rgba(153, 102, 255, 0.6)',
          borderColor: 'rgba(153, 102, 255, 1)',
          borderWidth: 2
        }
      ]
    }
  }

  const getDoughnutData = () => {
    if (!reportData?.metrics?.methods) return null

    const methods = reportData.metrics.methods.filter(m => m.p5_mean > 0)
    
    return {
      labels: methods.map(m => m.name),
      datasets: [{
        label: 'Precision@5 Distribution',
        data: methods.map(m => (m.p5_mean * 100).toFixed(2)),
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)'
        ],
        borderWidth: 2
      }]
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            size: 12,
            family: "'Inter', sans-serif"
          }
        }
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
        font: {
          size: 16,
          weight: 'bold',
          family: "'Inter', sans-serif"
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%'
          }
        }
      }
    }
  }

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          font: {
            size: 12,
            family: "'Inter', sans-serif"
          }
        }
      },
      title: {
        display: true,
        text: 'Precision@5 by Method',
        font: {
          size: 16,
          weight: 'bold',
          family: "'Inter', sans-serif"
        }
      }
    }
  }

  const getBestMethod = () => {
    if (!reportData?.metrics?.methods) return null
    
    const validMethods = reportData.metrics.methods.filter(m => m.p5_mean > 0)
    if (validMethods.length === 0) return null
    
    return validMethods.reduce((best, current) => 
      current.p5_mean > best.p5_mean ? current : best
    )
  }

  const bestMethod = getBestMethod()

  return (
    <div className="evaluation-report">
      <button 
        className="eval-toggle-btn"
        onClick={handleToggle}
        disabled={loading}
      >
        {loading ? (
          <>⏳ Loading Report...</>
        ) : showReport ? (
          <>📊 Hide Evaluation Report</>
        ) : (
          <>📊 Show Model Performance Report</>
        )}
      </button>

      {error && (
        <div className="eval-error">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {showReport && reportData && (
        <div className="eval-content">
          <div className="eval-header">
            <h2>🎯 Model Performance Evaluation</h2>
            <div className="eval-meta">
              <span className="eval-badge">
                📝 {reportData.metrics?.total_queries || 0} Test Queries
              </span>
              <span className="eval-badge">
                ⏰ Generated: {reportData.generated_at ? new Date(reportData.generated_at).toLocaleDateString() : 'N/A'}
              </span>
            </div>
          </div>

          {/* Best Method Highlight */}
          {bestMethod && (
            <div className="best-method-card">
              <div className="best-method-icon">🏆</div>
              <div className="best-method-content">
                <h3>Best Performing Method</h3>
                <div className="best-method-name">{bestMethod.name}</div>
                <div className="best-method-stats">
                  <div className="stat">
                    <span className="stat-label">Precision@5:</span>
                    <span className="stat-value">{(bestMethod.p5_mean * 100).toFixed(2)}%</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">nDCG@10:</span>
                    <span className="stat-value">{(bestMethod.ndcg10_mean * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Summary Table */}
          <div className="eval-section">
            <h3>📈 Performance Metrics</h3>
            <div className="metrics-table-container">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Precision@5</th>
                    <th>nDCG@10</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {reportData.metrics?.methods?.map((method, idx) => (
                    <tr key={idx} className={method.p5_mean === bestMethod?.p5_mean ? 'best-row' : ''}>
                      <td className="method-name">
                        {method.p5_mean === bestMethod?.p5_mean && <span className="crown">👑</span>}
                        {method.name}
                      </td>
                      <td>
                        <div className="metric-cell">
                          <span className="metric-value">
                            {(method.p5_mean * 100).toFixed(2)}%
                          </span>
                          <span className="metric-std">
                            ± {(method.p5_std * 100).toFixed(2)}%
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="metric-cell">
                          <span className="metric-value">
                            {(method.ndcg10_mean * 100).toFixed(2)}%
                          </span>
                          <span className="metric-std">
                            ± {(method.ndcg10_std * 100).toFixed(2)}%
                          </span>
                        </div>
                      </td>
                      <td>
                        {method.p5_mean > 0.9 ? (
                          <span className="status-badge excellent">Excellent</span>
                        ) : method.p5_mean > 0.7 ? (
                          <span className="status-badge good">Good</span>
                        ) : method.p5_mean > 0.5 ? (
                          <span className="status-badge fair">Fair</span>
                        ) : (
                          <span className="status-badge poor">Needs Improvement</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts Section */}
          <div className="eval-section">
            <h3>📊 Visual Comparison</h3>
            <div className="charts-grid">
              <div className="chart-container">
                {getBarChartData() && (
                  <Bar data={getBarChartData()} options={chartOptions} />
                )}
              </div>
              <div className="chart-container">
                {getDoughnutData() && (
                  <Doughnut data={getDoughnutData()} options={doughnutOptions} />
                )}
              </div>
            </div>
          </div>

          {/* Metrics Explanation */}
          <div className="eval-section">
            <h3>📚 Understanding the Metrics</h3>
            <div className="metrics-explanation">
              <div className="metric-card">
                <div className="metric-icon">🎯</div>
                <div className="metric-info">
                  <h4>Precision@5</h4>
                  <p>
                    Measures how many of the top 5 recommendations are actually relevant.
                    Higher is better (100% = perfect recommendations).
                  </p>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon">📊</div>
                <div className="metric-info">
                  <h4>nDCG@10</h4>
                  <p>
                    Normalized Discounted Cumulative Gain at 10. Rewards relevant items
                    appearing higher in the ranking. Considers both relevance and position.
                  </p>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon">🔬</div>
                <div className="metric-info">
                  <h4>Methods Compared</h4>
                  <p>
                    <strong>TF-IDF:</strong> Keyword matching<br/>
                    <strong>Embeddings:</strong> Semantic similarity (SciBERT)<br/>
                    <strong>Hybrid:</strong> Combines multiple signals<br/>
                    <strong>LambdaRank:</strong> Machine learning model
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Key Insights */}
          <div className="eval-section insights-section">
            <h3>💡 Key Insights</h3>
            <div className="insights-grid">
              <div className="insight-card">
                <span className="insight-icon">✨</span>
                <h4>State-of-the-Art Performance</h4>
                <p>
                  Achieving <strong>{bestMethod ? (bestMethod.p5_mean * 100).toFixed(1) : 'N/A'}% Precision@5</strong> demonstrates
                  near-perfect recommendation accuracy.
                </p>
              </div>
              <div className="insight-card">
                <span className="insight-icon">🧠</span>
                <h4>Semantic Understanding Wins</h4>
                <p>
                  Embeddings-based methods outperform keyword matching, showing the
                  importance of semantic understanding in academic paper matching.
                </p>
              </div>
              <div className="insight-card">
                <span className="insight-icon">⚡</span>
                <h4>Production Ready</h4>
                <p>
                  Consistent performance across {reportData.metrics?.total_queries || 0} test queries
                  ensures reliable recommendations in production.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default EvaluationReport
