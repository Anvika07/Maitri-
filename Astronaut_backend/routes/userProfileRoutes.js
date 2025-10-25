import express from 'express';
import { requireAuth } from '../middleware/authMiddleware.js';
import { getHealthStatus, switchPersona } from '../controllers/userProfileController.js';

const router = express.Router();

// All routes in this file are protected
router.use(requireAuth);

router.get('/health', getHealthStatus);
router.put('/persona', switchPersona);

export default router;
