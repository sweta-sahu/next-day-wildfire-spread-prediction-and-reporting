import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import TitleBar from '../components/TitleBar'
import ProcessFlow from '../components/ProcessFlow'
import AboutUs from '../components/AboutUs'
import ImageUpload from '../components/ImageUpload'

const Home: React.FC = () => {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const res = await fetch('http://localhost:8000/report/pdf/', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || res.statusText)
      }

      // read PDF as blob → create URL
      const blob = await res.blob()
      const reportUrl = URL.createObjectURL(blob)

      // navigate to /report and pass URL in location.state
      navigate('/results', { state: { reportUrl } })
    } catch (err: any) {
      console.error(err)
      setError(err.message || 'Failed to generate report')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-container">
      <TitleBar title="Wildfire Analysis System" />

      <div className="panel-container">
        <div className="process-flow"><ProcessFlow /></div>
        <div className="about-us"><AboutUs /></div>
      </div>

      <div className="upload-container">
        <ImageUpload
          onFileSelect={(file) => {
            setSelectedFile(file)
            setError(null)
          }}
        />

        {selectedFile && (
          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading ? 'Generating…' : 'Analyze Image'}
          </button>
        )}

        {error && (
          <p className="error" style={{ color: 'red' }}>{error}</p>
        )}
      </div>
    </div>
  )
}

export default Home
