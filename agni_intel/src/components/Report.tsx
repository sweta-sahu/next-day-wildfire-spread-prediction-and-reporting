import React from 'react'
import { ReportData } from '../types/ReportData';

interface ReportProps {
  data: ReportData;
}
const Report: React.FC<ReportProps> = ({ data }) => {
  const { title, reportDate, metrics, hotspots, actionItems, analysisImages } = data;

  return (
    <div className="report-container">
    <h2 className="mb-4 text-center">{title}</h2>
    <p className="text-center text-muted mb-5">
      Generated on: {new Date(reportDate).toLocaleString()}
    </p>

    <div className="mb-4">
      <h4>Spread Metrics</h4>
      <table className="table table-bordered">
        <tbody>
          <tr>
            <th>Yesterday's Burn</th>
            <td>{metrics.yesterdayBurn}</td>
          </tr>
          <tr>
            <th>Predicted Burn</th>
            <td>{metrics.predictedBurn}</td>
          </tr>
          <tr>
            <th>Area Increase</th>
            <td>{metrics.areaIncrease}</td>
          </tr>
          <tr>
            <th>Percent Increase</th>
            <td>{metrics.percentIncrease}</td>
          </tr>
          <tr>
            <th>Average Spread Speed</th>
            <td>{metrics.avgSpeed}</td>
          </tr>
          <tr>
            <th>Direction</th>
            <td>{metrics.direction}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div className="mb-4">
      <h4>Detected Hotspots</h4>
      <table className="table table-striped table-bordered">
        <thead>
          <tr>
            <th>X</th>
            <th>Y</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {hotspots.map((h, index) => (
            <tr key={index}>
              <td>{h.x}</td>
              <td>{h.y}</td>
              <td>{h.confidence.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>

    <div className="mb-4">
      <h4>Actionable Recommendations</h4>
      <ul className="mt-2">
        {actionItems.map((item, index) => (
          <li key={index} className="mb-2">{item}</li>
        ))}
      </ul>
    </div>

    <div className="mb-4">
      <h4 className="text-center mb-3">Risk Analysis Visualization</h4>
      <div className="d-flex flex-column gap-3">
        {analysisImages.map((imgUrl, index) => (
          <img
            key={index}
            src={imgUrl}
            alt={`Analysis visualization ${index + 1}`}
            className="report-image"
          />
        ))}
      </div>
    </div>

    <div className="mt-4 p-3 bg-light rounded">
      <h5>Additional Information</h5>
      <p>
        This report was generated based on satellite and image analysis performed on{' '}
        {new Date(reportDate).toLocaleDateString()}. The system uses proprietary models
        analyzing vegetation, topography, and past fire data to predict fire behavior and risk.
      </p>
    </div>
  </div>
);
};

export default Report;
