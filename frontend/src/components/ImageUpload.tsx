import React, { useState, useRef } from 'react'
import { UploadCloudIcon, FileIcon } from 'lucide-react'

interface ImageUploadProps {
	onFileSelect: (file: File) => void
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onFileSelect }) => {
	const [isDragging, setIsDragging] = useState(false)
	const [previewImage, setPreviewImage] = useState<string | null>(null)
	const [fileName, setFileName] = useState<string | null>(null)
	const fileInputRef = useRef<HTMLInputElement>(null)

  	const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
  		e.preventDefault()
  		setIsDragging(true)
  	}

  	const handleDragLeave = () => {
  	  	setIsDragging(false)
  	}

  	const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
  	  	e.preventDefault()
  	  	setIsDragging(false)
  	  	if (e.dataTransfer.files && e.dataTransfer.files[0]) {
  	  		processFile(e.dataTransfer.files[0])
  	  	}
  	}

  	const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
  	  	if (e.target.files && e.target.files[0]) {
  	  	  	processFile(e.target.files[0])
  	  	}
  	}

  	const processFile = (file: File) => {
    	setFileName(file.name)

    	const fileExt = file.name.toLowerCase()
    	if (!file.type.match('image.*') && !fileExt.endsWith('.tif') && !fileExt.endsWith('.tiff')) {
    		alert('Please select a valid image file')
    		return
    	}

    	onFileSelect(file) 

    	if (fileExt.endsWith('.tif') || fileExt.endsWith('.tiff')) {
    	  	setPreviewImage(null)
    	  	return
    	}

    	const reader = new FileReader()
    	reader.onload = (e) => {
    		if (e.target && typeof e.target.result === 'string') {
    	    	setPreviewImage(e.target.result)
    	  	}
    	}
    	reader.readAsDataURL(file)
  	}

  	const handleBoxClick = () => {
  		if (fileInputRef.current) {
  	    	fileInputRef.current.click()
  	  	}
  	}

  return (
		<div className="w-100 d-flex flex-column align-items-center">
      		<h3 className="mb-4">Upload Image for Analysis</h3>
      		<div
      			className={`upload-box ${isDragging ? 'border-primary' : ''} ${previewImage || fileName ? 'has-image' : ''}`}
      			onDragOver={handleDragOver}
      			onDragLeave={handleDragLeave}
      			onDrop={handleDrop}
      			onClick={handleBoxClick}
      		>
        		{fileName?.toLowerCase().endsWith('.tif') || fileName?.toLowerCase().endsWith('.tiff') ? (
          			<div className="text-center">
          			  <FileIcon size={40} className="mb-2" />
          			  <p>{fileName}</p>
          			</div>
        		) : 
				previewImage ? (
        		  	<div className="preview-container">
        		  		<img src={previewImage} alt="Preview" className="preview-image" />
        		  	</div>
        		) : 
				(
          		<>
          		  	<UploadCloudIcon size={48} className="mb-3" color="#E88C7D" />
          		  	<h4>Drag & Drop Image Here</h4>
          		  	<p className="text-muted">or click to browse files</p>
          		</>
        		)}
        		<input
        		  type="file"
        		  ref={fileInputRef}
        		  onChange={handleFileSelect}
        		  accept="image/*,.tif,.tiff"
        		  className="d-none"
        		/>
      		</div>
      		<p className="mt-3 text-muted">
      			Upload satellite images, aerial photographs, or TIFF files for wildfire risk assessment
      		</p>
		</div>
  )
}

export default ImageUpload;
