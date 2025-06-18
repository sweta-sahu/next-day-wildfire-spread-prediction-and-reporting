import React, { useEffect, useState } from 'react'
import {  Upload, Cpu, AlertTriangle, FileText } from 'lucide-react'
const ProcessFlow: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(true)
  const steps = [
    {
      icon: <Upload size={20} />,
      title: 'Ingest',
      description: 'User Uploads Satellite Image of WildFire',
    },
    {
      icon: <Cpu size={20} />,
      title: 'Infer',
      description: 'Model Predicts Next-Day Fire Mask',
    },
    {
      icon: <AlertTriangle size={20} />,
      title: 'Post-Processing',
      description: 'Model Derives Key Insights',
    },
    {
      icon: <FileText size={20} />,
      title: 'Narrate',
      description: 'LLM Produces a Concise, Actionable Report',
    },
    {
        icon: <FileText size={20} />,
        title: 'Deliver',
        description: 'Report used by Emergency Personnel',
      },
  ]
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (isAutoPlaying) {
      interval = setInterval(() => {
        setActiveStep((prev) => (prev + 1) % steps.length)
      }, 2000)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isAutoPlaying, steps.length])
  return (
    <div className="process-container">
      <h3 className="text-center mb-4">Wildfire Analysis Process</h3>
      <div className="process-steps">
        {steps.map((step, index) => (
          <div
            key={index}
            className={`process-step-box ${index === activeStep ? 'active' : ''} ${index < activeStep ? 'completed' : ''}`}
            onClick={() => {
              setIsAutoPlaying(false)
              setActiveStep(index)
            }}
          >
            <div className="step-icon">{step.icon}</div>
            <div className="step-content">
              <h4>{step.title}</h4>
              <p>{step.description}</p>
            </div>
            <div className="step-connector" />
          </div>
        ))}
      </div>
    </div>
  )
}
export default ProcessFlow;
