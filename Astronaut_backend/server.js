import express from 'express';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import path from 'path';
import { fileURLToPath } from 'url';
import { Server } from 'socket.io';
import { createServer } from 'http';
import { initializeSocket } from './socketHandler.js';

// Load environment variables as early as possible
// Resolve .env relative to this file so running nodemon from the repo root still finds it
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.join(__dirname, '.env') });

const app = express();
const httpServer = createServer(app); 
const io = new Server(httpServer, {
    cors: {
        origin: "*", // Allow all origins for now. In production, restrict this to your frontend URL.
        methods: ["GET", "POST"]
    }
});

const port = process.env.PORT || 3000;
app.use(express.json());

app.get('/', (req, res) => {
    res.send('Hello World');
});

// Mount auth routes
import authRoutes from './routes/authRoutes.js';
import userProfileRoutes from './routes/userProfileRoutes.js';
import { requireAuth } from './middleware/authMiddleware.js';
app.use('/api/auth', authRoutes);
app.use('/api/profile', userProfileRoutes);

// Protected test route
app.get('/api/protected', requireAuth, (req, res) => {
  res.json({ message: 'You accessed a protected route', user: req.user });
});

// Initialize Socket.IO logic
initializeSocket(io);

const mongoUrl = process.env.MONGODB_URL;
if (!mongoUrl) {
    console.error('MONGODB_URL is not defined in environment variables. Set it in your .env file.');
} else if (!/^mongodb(\+srv)?:\/\//i.test(mongoUrl)) {
    console.error('Invalid MONGODB_URL. It must start with "mongodb://" or "mongodb+srv://"');
}


async function start() {
    try {
        if (mongoUrl) {
            await mongoose.connect(mongoUrl);
            console.log('Connected to MongoDB');
        } else {
            console.warn('Skipping MongoDB connection because MONGODB_URL is not set.');
        }

        httpServer.listen(port, () => {
            console.log(`Server is running on port ${port}`);
        });
    } catch (error) {
        console.error('Error connecting to MongoDB:', error); 
        
        process.exit(1);
    }
}

start();




