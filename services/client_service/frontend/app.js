import React, { useState } from 'react';
import './styles/styles.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [inputData, setInputData] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ inputData }),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  return (
    <div className="app-container">
      <h1>Stock Prediction Client</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
          placeholder="Enter stock data"
        />
        <button type="submit" disabled={!inputData}>
          Predict
        </button>
      </form>
      {prediction && (
        <div className="prediction-result">
          Prediction Result: {JSON.stringify(prediction)}
        </div>
      )}
    </div>
  );
}

export default App;

