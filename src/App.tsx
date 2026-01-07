import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DocumentCenter from './pages/DocumentCenter';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<DocumentCenter />} />
        <Route path="/documents" element={<DocumentCenter />} />
      </Routes>
    </Router>
  );
}

export default App
