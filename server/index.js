import express from 'express';
import cors from 'cors';

const app = express();
const port = process.env.PORT || 8080;

// ✅ Step 1: Enable CORS for your frontend origins
app.use(cors({
  origin: ['https://galcast.co', 'https://bolt.new'],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
  credentials: true
}));

// ✅ Step 2: Middleware for JSON parsing
app.use(express.json());

// ✅ Optional: Log incoming requests (for debugging)
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// ✅ Step 3: Main analysis endpoint
app.post('/analyze-portfolio', (req, res) => {
  const { stocks, dateRange } = req.body;

  if (!stocks || !Array.isArray(stocks) || !stocks.length) {
    return res.status(400).json({ success: false, error: 'Missing or invalid stocks data.' });
  }

  if (!dateRange || !dateRange.startDate || !dateRange.endDate) {
    return res.status(400).json({ success: false, error: 'Missing or invalid dateRange.' });
  }

  // Stub response — replace with real logic
  res.json({
    success: true,
    message: 'Portfolio analysis received.',
    received: {
      stocks,
      dateRange
    }
  });
});

// ✅ Step 4: Catch-all for unknown routes
app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Route not found.' });
});

// ✅ Step 5: Start the server
app.listen(port, () => {
  console.log(`✅ Server running on port ${port}`);
});
