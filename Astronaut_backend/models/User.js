import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true, lowercase: true, trim: true },
  password: { type: String, required: true },
  persona: { type: String, default: 'Student' }
}, { timestamps: true });

const User = mongoose.model('User', userSchema);
export default User;
