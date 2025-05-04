import React, { useRef, useState, useEffect } from 'react';
import {  useNavigate } from 'react-router-dom';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
// import html2canvas from 'html2canvas';
import TitleBar from '../components/TitleBar';
import Report from '../components/Report';
import { ReportData } from '../types/ReportData';

const Results: React.FC = () => {
  const navigate = useNavigate();
  const reportRef = useRef<HTMLDivElement>(null);
  const [reportData, setReportData] = useState<ReportData | null>(null);


  useEffect(() => {
    const fetchReport = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/report');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        setReportData(data);
      } catch (err) {
        console.error('Error fetching report data:', err);
        setReportData(null);
      }
    };
  
    fetchReport();
  
    
  }, []);
  
 

  const handleExportPDF = async () => {
    if (!reportData) return;

    const pdf = new jsPDF('p', 'mm', 'a4');
    const marginLeft = 15;
    let y = 20;

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const maxWidth = pageWidth - marginLeft * 2;

    pdf.setFontSize(20);
    pdf.setTextColor(0);
    pdf.text('Analysis Report', marginLeft, y);
    y += 12;

    pdf.setFontSize(12);
    pdf.setTextColor(80);
    const summary = "This report summarizes the predicted spread, critical hotspots, and key action items.";
    const summaryLines = pdf.splitTextToSize(summary, maxWidth);
    pdf.text(summaryLines, marginLeft, y);
    y += summaryLines.length * 6 + 6;

    pdf.setTextColor(0);
    pdf.setFontSize(14);
    pdf.text('Actionable Recommendations', marginLeft, y);
    y += 8;

    pdf.setFontSize(12);
    reportData.actionItems.forEach((item: string) => {
      const lines = pdf.splitTextToSize(`• ${item}`, maxWidth);
      pdf.text(lines, marginLeft, y);
      y += lines.length * 6;
    });

    y += 10;

    pdf.setFontSize(14);
    pdf.text('Risk Analysis Visualizations', marginLeft, y);
    y += 10;

    for (const imgUrl of reportData.analysisImages) {
      try {
        const imgData = await loadImageAsBase64(imgUrl);
        const imgProps = pdf.getImageProperties(imgData);
        const imgHeight = (imgProps.height * maxWidth) / imgProps.width;

        if (y + imgHeight > pageHeight - 20) {
          pdf.addPage();
          y = 20;
        }
        pdf.addImage(imgData, 'JPEG', marginLeft, y, maxWidth, imgHeight);
        y += imgHeight + 10;
      } catch (err) {
        console.error('Failed to load image', imgUrl, err);
      }
    }

    pdf.save('wildfire-report.pdf');
  };

  const loadImageAsBase64 = (url: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return reject(new Error('Could not get canvas context'));
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL('image/jpeg'));
      };
      img.onerror = reject;
      img.src = url;
    });
  };


  return (
    <div className="page-container">
      <TitleBar title="Analysis Results" />
      <div className="results-container">
      {reportData ? (
          <div ref={reportRef}>
            <Report data={reportData} />
          </div>
        ) : (
          <div className="text-center my-5 text-muted">No report data available.</div>
        )}
        <div className="d-flex justify-content-between mt-4">
          <button className="btn btn-secondary" onClick={() => navigate('/')}>
            Back to Home
          </button>
          <button
            className="btn export-btn"
            onClick={handleExportPDF}
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
