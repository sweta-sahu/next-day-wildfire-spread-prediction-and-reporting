import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import TitleBar from '../components/TitleBar';
import Report from '../components/Report';

const Results: React.FC = () => {
	const navigate = useNavigate();
	const location = useLocation();
	const [reportData, setReportData] = useState<any | null>(location.state?.reportData || null);
	const [error, setError] = useState<string | null>(null);
	
	useEffect(() => {
		if (!reportData) {
			fetch('/sample_report.json')
				.then((res) => {
			  	if (!res.ok) {
					throw new Error('Failed to load fallback report');
			  	}
			  	return res.json();
			})
			.then((data) => setReportData(data))
			.catch((err) => {
			  	console.error('Error loading fallback:', err);
			  	setError('No report data available.');
			});
		}
	}, [reportData]);


  return (
    	<div className="page-container">
      		<TitleBar title="Analysis Results" />
      		<div className="results-container">
      			{reportData ? (
          			<div>
          				<Report data={reportData} />
          			</div>
      			) : 
				(
					<div className="text-center my-5 text-muted">
					  	{error || 'Loading...'}
					</div>
				)}
        		<div className="d-flex justify-content-between mt-4">
          			<button className="btn btn-secondary" onClick={() => navigate('/')}>
          			  	Back to Home
          			</button>
          			<button
          			  	className="btn export-btn"
          			  	onClick={() => { /* no-op */ }}
          			  	disabled={!reportData}
          			>
          			  	Export as PDF
          			</button>
        		</div>
      		</div>
    	</div>
  );
};

export default Results;
