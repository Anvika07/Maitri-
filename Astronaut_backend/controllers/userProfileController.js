import User from '../models/User.js';
import EmotionLog from '../models/EmotionLog.js';

export async function getHealthStatus(req, res) {
    try {
        // Fetch recent emotion logs for the user
        const recentLogs = await EmotionLog.find({ userId: req.user.id })
            .sort({ createdAt: -1 })
            .limit(20);

        if (recentLogs.length === 0) {
            return res.json({
                fatigue: 'low',
                stress: 'low',
                mood: 'neutral',
                recentHistory: []
            });
        }

        // Simple logic to determine current stress/fatigue
        // This can be made much more sophisticated
        const highStressCount = recentLogs.filter(log => log.stressLevel === 'high').length;
        const mediumStressCount = recentLogs.filter(log => log.stressLevel === 'medium').length;

        let stress = 'low';
        if (highStressCount > 2) {
            stress = 'high';
        } else if (highStressCount > 0 || mediumStressCount > 3) {
            stress = 'medium';
        }

        const latestEmotion = recentLogs[0].emotion;

        res.json({
            fatigue: 'normal', // Could be calculated based on time of day, activity, etc.
            stress: stress,
            mood: latestEmotion,
            recentHistory: recentLogs.map(log => ({
                emotion: log.emotion,
                source: log.source,
                stress: log.stressLevel,
                timestamp: log.createdAt
            }))
        });
    } catch (error) {
        console.error('Error fetching health status:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}

export async function switchPersona(req, res) {
    const { persona } = req.body;
    if (!persona) {
        return res.status(400).json({ message: 'Persona is required' });
    }

    try {
        const user = await User.findByIdAndUpdate(req.user.id, { persona }, { new: true });
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        console.log(`User ${req.user.id} switched persona to ${persona}`);
        res.json({ message: `Persona updated to ${persona}`, persona: user.persona });
    } catch (error) {
        console.error('Error switching persona:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}
