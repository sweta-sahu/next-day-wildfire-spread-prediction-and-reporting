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

      // call for API
      const res = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) throw new Error(await res.text() || res.statusText)
      const data = await res.json()

      // Passing the JSON and the original file to Results
      navigate('/results', {
        state: { reportData: data, file: selectedFile },
      })
    } catch (err: any) {
      console.error('API / predict failed:', err)
      setError('Failed to analyze image.')
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
            {loading ? 'Analyzing…' : 'Analyze Image'}
          </button>
        )}

        {error && <p className="error text-danger">{error}</p>}
      </div>
    </div>
  )
}

export default Home
