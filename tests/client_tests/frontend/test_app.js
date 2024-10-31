import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import App from '../../../client_service/frontend/src/App';

// Mock fetch for API calls
global.fetch = jest.fn(() =>
  Promise.resolve({
    json: () => Promise.resolve({ prediction: "mocked_prediction" })
  })
);

test('renders the prediction input and button', () => {
  render(<App />);
  expect(screen.getByPlaceholderText(/Enter stock data/i)).toBeInTheDocument();
  expect(screen.getByText(/Predict/i)).toBeInTheDocument();
});

test('submits data and displays prediction', async () => {
  render(<App />);
  
  // Input text and click Predict
  fireEvent.change(screen.getByPlaceholderText(/Enter stock data/i), { target: { value: 'test_stock_data' } });
  fireEvent.click(screen.getByText(/Predict/i));

  // Expect the prediction to be displayed
  const prediction = await screen.findByText(/Prediction Result: {"prediction":"mocked_prediction"}/i);
  expect(prediction).toBeInTheDocument();
});

