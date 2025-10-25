import mongoose from 'mongoose';

const emotionLogSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    source: {
        type: String,
        enum: ['face', 'voice', 'text'],
        required: true
    },
    emotion: {
        type: String,
        required: true
    },
    stressLevel: {
        type: String,
        enum: ['low', 'medium', 'high'],
        default: 'low'
    },
    confidence: {
        type: Number
    },
}, { timestamps: true });

const EmotionLog = mongoose.model('EmotionLog', emotionLogSchema);

export default EmotionLog;
