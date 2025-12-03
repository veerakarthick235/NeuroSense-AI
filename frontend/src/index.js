import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // Assuming your global styles are here
import { Toaster } from 'sonner'; // Assuming 'sonner' for toasts as used in dashboards

// The main entry point for the React application
const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    {/* The main application component.
      It should wrap the routing logic (like React Router DOM) 
      and the AuthProvider (as indicated by useAuth in your components).
    */}
    <App /> 
    
    {/* Toaster component for displaying notifications/toasts */}
    <Toaster 
        position="bottom-right" 
        richColors 
        theme="dark" 
    />
  </React.StrictMode>
);

// If you are using React Router, the <BrowserRouter> (or similar router) 
// typically wraps <App /> inside the App.js component, or sometimes right here.
// Based on the 'useNavigate' hook used in your dashboards, the router is likely
// wrapped around the entire <App /> component.

// The `App` component must contain the AuthProvider 
// and the BrowserRouter to support the hooks used in the dashboards.

/*
// Example of alternative setup if BrowserRouter was here:
import { BrowserRouter } from 'react-router-dom';

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
      <Toaster position="bottom-right" richColors theme="dark" />
    </BrowserRouter>
  </React.StrictMode>
);
*/
