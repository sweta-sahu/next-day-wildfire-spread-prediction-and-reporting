import React from 'react'
const AboutUs: React.FC = () => {
  return (
      

      	
		<div 
		className='about_us_container'
		>
		<h3 
		className='text-center mb-4'
		> About Us</h3>
 		    	<li 
				className='mb-2'
				>
					Problem: Wildfires spread rapidly, endangering lives and property; current systems lack actionable next-day forecasts.</li>
 		    	<li 
				className='mb-2'
				>
					Solution: Use multi-channel satellite imagery to predict the wildfire spread for the next day using segmentation models.
				</li>
				<li 
				className='mb-2'
				>
					LLM Integration: Convert spatial predictions into human-readable reports using a Large Language Model.
				</li>
				<li 
				className='mb-2'
				>
					Impact: Equip emergency managers with early, structured insights to guide evacuations, alerts, and resource deployment.
				</li>
				<li 
				className='mb-2'
				>
					Innovation: Seamless end-to-end pipeline that fuses deep learning with natural language generation for real-world decision support.
				</li>
				
				</div>
 		    	
 		  	
 	
		
    
 
  )
}
export default AboutUs;
