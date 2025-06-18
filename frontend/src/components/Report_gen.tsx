import React from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

type LocationState = {
  reportUrl?: string
}

const Report: React.FC = () => {
  const { state } = useLocation() as { state: LocationState }
  const navigate = useNavigate()
  const reportUrl = state?.reportUrl

  if (!reportUrl) {
    return (
      <div style={{ padding: '2rem' }}>
        <p>No report to display.</p>
        <button className="analyze-btn" onClick={() => navigate('/')}>
          Back Home
        </button>
      </div>
    )
  }

  const handleDownload = () => {
    const a = document.createElement('a')
    a.href = reportUrl
    a.download = 'wildfire_report.pdf'
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  return (
    <div className="report-page" style={{ padding: '1rem' }}>
      <h3>Wildfire Incident Report</h3>
      <iframe
        src={reportUrl}
        title="Wildfire Report"
        width="100%"
        height="600px"
        style={{ border: '1px solid #ccc', marginTop: '1rem' }}
      />
      <button
        className="analyze-btn"
        onClick={handleDownload}
        style={{ marginTop: '1rem' }}
      >
        Download Report
      </button>
    </div>
  )
}

export default Report
