import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import TitleBar from '../components/TitleBar'
import ProcessFlow from '../components/ProcessFlow'
import AboutUs from '../components/AboutUs'
import ImageUpload from '../components/ImageUpload'
const Home: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const navigate = useNavigate()
  const handleImageUpload = (imageDataUrl: string) => {
    setSelectedImage(imageDataUrl)
  }
  const handleAnalyze = () => {
    if (selectedImage) {
      navigate('/results', {
        state: {
          imageDataUrl: selectedImage,
        },
      })
    }
  }
  return (
    <div className="page-container">
      <TitleBar title="Wildfire Analysis System" />
      <div className="panel-container"> 
        <div className="process-flow">
          <ProcessFlow />
        </div>
        <div className="about-us">
          <AboutUs />
        </div>
      </div>

      <div className="upload-container">
        <ImageUpload onImageUpload={handleImageUpload} />
        {selectedImage && (
          <button className="analyze-btn" onClick={handleAnalyze}>
            Analyze Image
          </button>
        )}
      </div>
    </div>
  )
}
export default Home;
