import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Brain, Mail, Lock, User, UserCheck } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { toast } from 'sonner';
import { useAuth } from '../App';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    role: 'patient'
  });
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register';
      const response = await axios.post(`${API}${endpoint}`, formData);
      
      login(response.data.token, response.data.user);
      toast.success(isLogin ? 'Welcome back!' : 'Account created successfully!');
      
      const redirectPath = response.data.user.role === 'patient' ? '/dashboard' : '/research';
      navigate(redirectPath);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div data-testid="auth-page" className="min-h-screen bg-[#030712] flex items-center justify-center px-6 py-12">
      <div className="absolute inset-0 opacity-20">
        <img 
          src="https://images.unsplash.com/photo-1764336312138-14a5368a6cd3?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzl8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMG5ldXJhbCUyMG5ldHdvcmslMjBnbG93aW5nJTIwYmx1ZSUyMHB1cnBsZXxlbnwwfHx8fDE3NjQ1OTM0OTB8MA&ixlib=rb-4.1.0&q=85"
          alt="Neural network"
          className="w-full h-full object-cover"
        />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md relative z-10"
      >
        <div className="glass-card p-8 rounded-2xl">
          {/* Logo */}
          <div className="flex items-center justify-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-[#00F0FF] to-[#7C3AED] flex items-center justify-center">
              <Brain className="w-7 h-7 text-black" />
            </div>
            <span className="text-2xl font-bold font-['Outfit']">NeuroSense AI</span>
          </div>

          {/* Title */}
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 font-['Outfit']">
              {isLogin ? 'Welcome Back' : 'Create Account'}
            </h2>
            <p className="text-white/60">
              {isLogin ? 'Sign in to your account' : 'Start monitoring your cognitive health'}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <Label htmlFor="name" className="text-white/80 mb-2 block">
                  <User className="w-4 h-4 inline mr-2" />
                  Full Name
                </Label>
                <Input
                  data-testid="auth-name-input"
                  id="name"
                  type="text"
                  placeholder="John Doe"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required={!isLogin}
                  className="bg-black/50 border-white/20 focus:border-[#00F0FF] text-white"
                />
              </div>
            )}

            <div>
              <Label htmlFor="email" className="text-white/80 mb-2 block">
                <Mail className="w-4 h-4 inline mr-2" />
                Email Address
              </Label>
              <Input
                data-testid="auth-email-input"
                id="email"
                type="email"
                placeholder="you@example.com"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                required
                className="bg-black/50 border-white/20 focus:border-[#00F0FF] text-white"
              />
            </div>

            <div>
              <Label htmlFor="password" className="text-white/80 mb-2 block">
                <Lock className="w-4 h-4 inline mr-2" />
                Password
              </Label>
              <Input
                data-testid="auth-password-input"
                id="password"
                type="password"
                placeholder="••••••••"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                required
                className="bg-black/50 border-white/20 focus:border-[#00F0FF] text-white"
              />
            </div>

            {!isLogin && (
              <div>
                <Label className="text-white/80 mb-3 block">
                  <UserCheck className="w-4 h-4 inline mr-2" />
                  Account Type
                </Label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    data-testid="role-patient-btn"
                    type="button"
                    onClick={() => setFormData({ ...formData, role: 'patient' })}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      formData.role === 'patient'
                        ? 'border-[#00F0FF] bg-[#00F0FF]/10'
                        : 'border-white/10 bg-black/30 hover:border-white/20'
                    }`}
                  >
                    <div className="text-center">
                      <User className="w-6 h-6 mx-auto mb-2" />
                      <div className="font-medium">Patient</div>
                    </div>
                  </button>
                  <button
                    data-testid="role-researcher-btn"
                    type="button"
                    onClick={() => setFormData({ ...formData, role: 'researcher' })}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      formData.role === 'researcher'
                        ? 'border-[#00F0FF] bg-[#00F0FF]/10'
                        : 'border-white/10 bg-black/30 hover:border-white/20'
                    }`}
                  >
                    <div className="text-center">
                      <UserCheck className="w-6 h-6 mx-auto mb-2" />
                      <div className="font-medium">Researcher</div>
                    </div>
                  </button>
                </div>
              </div>
            )}

            <Button
              data-testid="auth-submit-btn"
              type="submit"
              disabled={loading}
              className="w-full bg-[#00F0FF] text-black font-bold hover:shadow-[0_0_30px_rgba(0,240,255,0.5)] transition-all py-6 text-lg"
            >
              {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
            </Button>
          </form>

          {/* Toggle */}
          <div className="mt-6 text-center">
            <button
              data-testid="auth-toggle-btn"
              onClick={() => setIsLogin(!isLogin)}
              className="text-white/60 hover:text-[#00F0FF] transition-colors"
            >
              {isLogin ? "Don't have an account? " : 'Already have an account? '}
              <span className="font-bold text-[#00F0FF]">
                {isLogin ? 'Sign Up' : 'Sign In'}
              </span>
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AuthPage;
