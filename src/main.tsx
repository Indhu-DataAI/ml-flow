
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { Routes, Route, BrowserRouter } from 'react-router-dom';
import React from 'react';
import Dashboard from './components/Dashboard.tsx';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/dashboard" element={<Dashboard />} /> 
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);


