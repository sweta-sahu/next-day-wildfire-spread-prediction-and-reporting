import React from 'react'
import { FlameIcon } from 'lucide-react'
interface TitleBarProps {
  title: string
}
const TitleBar: React.FC<TitleBarProps> = ({ title }) => {
  return (
    <div className="title-bar d-flex align-items-center">
      <FlameIcon size={32} className="me-3" />
      <h1 className="m-0">{title}</h1>
    </div>
  )
}
export default TitleBar;
