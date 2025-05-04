// src/pages/Results.tsx
import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Document, Page, pdfjs } from 'react-pdf'
import TitleBar from '../components/TitleBar'
import 'react-pdf/dist/esm/Page/AnnotationLayer.css'

// point at CDN worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

type LocationState = { reportUrl?: string }

const Results: React.FC = () => {
  const navigate = useNavigate()
  const { state } = useLocation() as { state: LocationState }
  const reportUrl = state?.reportUrl

  const [numPages, setNumPages]     = useState(0)
  const [pageNumber, setPageNumber] = useState(1)
  const [scale, setScale]           = useState(1.0)

  if (!reportUrl) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-600 mb-4">No report to display.</p>
          <button
            className="analyze-btn"
            onClick={() => navigate('/')}
          >
            Back to Home
          </button>
        </div>
      </div>
    )
  }

  const onDocumentLoadSuccess = (pdf: any) => {
    setNumPages(pdf.numPages)
    setPageNumber(1)
  }
  const changePage = (offset: number) =>
    setPageNumber(p => Math.min(Math.max(p + offset, 1), numPages))
  const downloadReport = () => {
    const a = document.createElement('a')
    a.href = reportUrl
    a.download = 'wildfire_report.pdf'
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <TitleBar title="Wildfire Incident Report" />

      {/* Toolbar */}
      <div className="sticky top-0 z-20 flex items-center bg-white shadow px-6 py-3">
        <button
          className="analyze-btn mr-2"
          onClick={() => changePage(-1)}
          disabled={pageNumber <= 1}
        >
          ‹ Prev
        </button>
        <span className="text-gray-700 mx-2">
          Page {pageNumber} / {numPages}
        </span>
        <button
          className="analyze-btn mr-auto"
          onClick={() => changePage(1)}
          disabled={pageNumber >= numPages}
        >
          Next ›
        </button>

        <button
          className="analyze-btn mr-2"
          onClick={() => setScale(s => Math.max(0.4, s - 0.2))}
          disabled={scale <= 0.4}
        >
          – Zoom
        </button>
        <button
          className="analyze-btn mr-6"
          onClick={() => setScale(s => Math.min(3, s + 0.2))}
          disabled={scale >= 3}
        >
          + Zoom
        </button>

        <button
          className="analyze-btn"
          onClick={downloadReport}
        >
          Download
        </button>
      </div>

      {/* PDF Viewer */}
      <div className="flex-1 overflow-auto p-6">
        <div className="mx-auto max-w-4xl bg-white rounded-lg shadow-lg overflow-hidden">
          <Document
            file={reportUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={<p className="p-6 text-center">Loading report…</p>}
            noData={<p className="p-6 text-center">No PDF file specified.</p>}
          >
            <Page
              pageNumber={pageNumber}
              scale={scale}
              loading={<p className="p-6 text-center">Rendering page…</p>}
            />
          </Document>
        </div>
      </div>
    </div>
  )
}

export default Results
