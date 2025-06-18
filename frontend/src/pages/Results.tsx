// src/pages/Results.tsx
import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import TitleBar from '../components/TitleBar'
import Report from '../components/Report'

interface LocationState {
  reportData: any
  file: File
}

const Results: React.FC = () => {
  const navigate = useNavigate()
  const { state } = useLocation() as { state?: LocationState }
  const [downloading, setDownloading] = useState(false)

  if (!state?.reportData || !state.file) {
    return (
      <div className="page-container p-5 text-center">
        <TitleBar title="Analysis Results" />
        <p>No report data found. Please run an analysis first.</p>
        <button className="analyze-btn" onClick={() => navigate('/')}>
          Back to Home
        </button>
      </div>
    )
  }

  const { reportData, file } = state

  const handleDownload = async () => {
    setDownloading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('http://localhost:8000/report/pdf/', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) throw new Error(await res.text() || res.statusText)

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'wildfire_report.pdf'
      document.body.appendChild(a)
      a.click()
      a.remove()
    } catch (err) {
      console.error('API /report/pdf failed:', err)
      alert('Failed to download report.')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="page-container">
      <TitleBar title="Analysis Results" />

      <div className="results-container">
        <Report data={reportData} />

        <div className="d-flex justify-content-between mt-4">
          <button
            className="analyze-btn"
            onClick={() => navigate('/')}
          >
            Back to Home
          </button>
          <button
            className="analyze-btn"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading ? 'Downloading…' : 'Download Report'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default Results
