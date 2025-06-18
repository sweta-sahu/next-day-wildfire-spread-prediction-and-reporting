import React, { useEffect, useState, Fragment } from 'react'
const Graph: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(true)
  const steps = [
    'Upload Image',
    'Process Data',
    'Analyze Risks',
    'Generate Report',
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
  const handleStepClick = (index: number) => {
    setIsAutoPlaying(false)
    setActiveStep(index)
  }
  const handleCenterClick = () => {
    setIsAutoPlaying((prev) => !prev)
  }
  const radius = 120
  const center = 150
  return (
    <div className="circular-process-container">
      <h3 className="mb-4 text-center">Wildfire Analysis Process</h3>
      <svg width="300" height="300" viewBox="0 0 300 300">
        {/* Center circle */}
        <circle
          cx={center}
          cy={center}
          r={50}
          fill="#E25822"
          opacity="0.9"
          style={{
            cursor: 'pointer',
          }}
          onClick={handleCenterClick}
        />
        <text
          x={center}
          y={center - 10}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="white"
          fontSize="14"
          fontWeight="bold"
        >
          WILDFIRE
        </text>
        <text
          x={center}
          y={center + 10}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="white"
          fontSize="10"
        >
          {isAutoPlaying ? 'Click to pause' : 'Click to play'}
        </text>
        {/* Process steps */}
        {steps.map((step, index) => {
          const angle = (index * 2 * Math.PI) / steps.length - Math.PI / 2
          const x = center + radius * Math.cos(angle)
          const y = center + radius * Math.sin(angle)
          const innerRadius = 60
          const innerX = center + innerRadius * Math.cos(angle)
          const innerY = center + innerRadius * Math.sin(angle)
          return (
            <Fragment key={index}>
              <line
                x1={innerX}
                y1={innerY}
                x2={x}
                y2={y}
                className={`process-line ${index === activeStep ? 'active' : ''}`}
              />
              <circle
                cx={x}
                cy={y}
                r={30}
                className={`process-step ${index === activeStep ? 'active' : ''}`}
                onClick={() => handleStepClick(index)}
              />
              <text x={x} y={y} className="process-text">
                {index + 1}
              </text>
              <text
                x={x}
                y={y + 45}
                textAnchor="middle"
                fill="#333"
                fontSize="12"
                fontWeight={index === activeStep ? 'bold' : 'normal'}
              >
                {step}
              </text>
            </Fragment>
          )
        })}
      </svg>
      <div className="mt-4 text-center">
        <p className="text-muted">
          {steps[activeStep]}: {getStepDescription(activeStep)}
        </p>
      </div>
    </div>
  )
}
const getStepDescription = (step: number): string => {
  switch (step) {
    case 0:
      return 'Upload your satellite or aerial imagery for analysis'
    case 1:
      return 'Advanced algorithms process the image data'
    case 2:
      return 'AI identifies potential wildfire risks'
    case 3:
      return 'Comprehensive report generation with recommendations'
    default:
      return ''
  }
}
export default Graph;
