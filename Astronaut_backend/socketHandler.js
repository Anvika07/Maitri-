import jwt from 'jsonwebtoken';
import axios from 'axios';
import EmotionLog from './models/EmotionLog.js';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export function initializeSocket(io) {
    // Middleware to authenticate socket connections
    io.use((socket, next) => {
        const token = socket.handshake.auth.token;
        if (!token) {
            return next(new Error('Authentication error: Token not provided.'));
        }
        jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
            if (err) {
                return next(new Error('Authentication error: Invalid token.'));
            }
            socket.user = decoded; // Attach user payload to the socket
            next();
        });
    });

    io.on('connection', (socket) => {
        console.log(`User connected: ${socket.user.email} (ID: ${socket.id})`);

        // Listener for facial analysis stream
        socket.on('analyze-face', async (frameBuffer) => {
            try {
                // TODO: Forward frameBuffer to Python API
                // const response = await axios.post(`${PYTHON_API_URL}/detect_emotion_face`, frameBuffer, {
                //     headers: { 'Content-Type': 'application/octet-stream' }
                // });
                // const modelResponse = response.data;

                // Using mock response for now
                const modelResponse = { emotion: 'happy', confidence: 0.92, stressLevel: 'low' };

                await EmotionLog.create({
                    userId: socket.user.id,
                    source: 'face',
                    emotion: modelResponse.emotion,
                    stressLevel: modelResponse.stressLevel,
                    confidence: modelResponse.confidence
                });

                // Send result back to the client
                socket.emit('analysis-result', { ...modelResponse, source: 'face' });

            } catch (error) {
                console.error('Face analysis error:', error.message);
                socket.emit('analysis-error', { source: 'face', message: 'Failed to analyze facial data.' });
            }
        });

        // Listener for voice analysis stream
        socket.on('analyze-voice', async (audioBuffer) => {
            try {
                // TODO: Forward audioBuffer to Python API
                const modelResponse = { emotion: 'calm', confidence: 0.88, stressLevel: 'low' };

                await EmotionLog.create({
                    userId: socket.user.id,
                    source: 'voice',
                    emotion: modelResponse.emotion,
                    stressLevel: modelResponse.stressLevel,
                    confidence: modelResponse.confidence
                });

                socket.emit('analysis-result', { ...modelResponse, source: 'voice' });

            } catch (error) {
                console.error('Voice analysis error:', error.message);
                socket.emit('analysis-error', { source: 'voice', message: 'Failed to analyze voice data.' });
            }
        });

        socket.on('disconnect', () => {
            console.log(`User disconnected: ${socket.user.email}`);
        });
    });
}
