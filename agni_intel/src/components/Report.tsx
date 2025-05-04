import React from 'react'

interface ReportProps {
	data: any;
}
const Report: React.FC<ReportProps> = ({ data }) => {
	const generatedAt = new Date().toLocaleString()
  	return (
    	<div className="report-container">
    		<h2 className="mb-4 text-center">Wildfire Incident Report</h2>
    		<p className="text-center text-muted mb-5">
      			Generated on:  {generatedAt}
    		</p>

    		<div className="d-flex flex-wrap justify-content-between mb-4 gap-4">
  				<div className="report-panel">
  				  	<h4 className="text-center">Spread Metrics</h4>
  				  	<table className="custom-table">
  				    	<tbody>
  				    	  	<tr><th>Yesterday's Burn</th><td>{data.dayt_true_area_km2.toFixed(2)} km²</td></tr>
  				    	  	<tr><th>Predicted Burn</th><td>{data.dayt1_pred_area_km2.toFixed(2)} km²</td></tr>
  				    	  	<tr><th>Area Increase</th><td>{data.delta_area_km2.toFixed(2)} km²</td></tr>
  				    	  	<tr><th>Spread Direction</th><td>{data.direction || 'N/A'}</td></tr>
  				    	</tbody>
  				  </table>
  				</div>

  				<div className="report-panel">
  				  	<h4 className="text-center">Detected Hotspots</h4>
  				  	<table className="custom-table">
  				  	  	<thead>
  				  	  	  	<tr><th>#</th><th>X</th><th>Y</th><th>Confidence</th></tr>
  				  	  	</thead>
  				  	  	<tbody>
  				  	  	  	{data.hotspots.map((h: any, i: number) => (
  				  	  	    	<tr key={i}><td>{i + 1}</td><td>{h.x}</td><td>{h.y}</td><td>{h.p.toFixed(3)}</td></tr>
  				  	  	  	))}
  				  	  	</tbody>
  				  	</table>
  				</div>
			</div>

			<div className="d-flex flex-wrap justify-content-between mb-4 gap-4">
			  	<div className="image-wrapper">
			    	<div>
			      		<h4 className="text-center">Predicted Next-Day Fire</h4>
			      		<img src={data.rgb_pred_png} alt="Predicted Fire Map" className="report-image" />
			    	</div>
			  	</div>

			  	<div className="image-wrapper">
			    	<div>
			      		<h4 className="text-center">Burn Corridor</h4>
			      		<img src={data.corridor_rgb_png} alt="Burn Corridor" className="report-image" />
			    	</div>
			  	</div>
			</div>

  		</div>
	);
};

export default Report;
