import express from 'express';
import cors from 'cors';

const app = express();
const port = process.env.PORT || 8080;

// ✅ Step 1: Allow frontend origins (Bolt and galcast.co)
app.use(cors({
  origin: ['https://galcast.co', 'https://bolt.new'],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
  credentials: true
}));

app.use(express.json());

// ✅ Step 2: Define your API endpoint
app.post('/analyze-portfolio', (req, res) => {
  // Example logic — replace with real analysis logic
  const { stocks, dateRange } = req.body;
  if (!stocks || !dateRange) {
    return res.status(400).json({ error: 'Missing data' });
  }
  res.json({ success: true, message: 'Portfolio analysis stub.' });
});

// ✅ Step 3: Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
