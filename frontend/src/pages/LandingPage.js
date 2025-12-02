import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Activity, Users, TrendingUp, Shield, Zap } from 'lucide-react';
import { Button } from '../components/ui/button';
import { useAuth } from '../App';

const LandingPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const handleGetStarted = () => {
    if (user) {
      navigate(user.role === 'patient' ? '/dashboard' : '/research');
    } else {
      navigate('/auth');
    }
  };

  const features = [
    {
      icon: Brain,
      title: 'Neural Pattern Analysis',
      description: 'Advanced AI-powered analysis of vocal patterns and cognitive markers'
    },
    {
      icon: Activity,
      title: 'Movement Dynamics',
      description: 'Real-time tracking of gait stability and motor function patterns'
    },
    {
      icon: Users,
      title: 'Social Interaction',
      description: 'Monitor engagement levels and communication patterns'
    },
    {
      icon: TrendingUp,
      title: 'Predictive Insights',
      description: 'AI-generated health insights and trend analysis'
    },
    {
      icon: Shield,
      title: 'TBI Risk Assessment',
      description: 'Early detection of traumatic brain injury risk factors'
    },
    {
      icon: Zap,
      title: 'Real-time Monitoring',
      description: 'Continuous multimodal data fusion and analysis'
    }
  ];

  return (
    <div data-testid="landing-page" className="min-h-screen bg-[#030712] text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00F0FF] to-[#7C3AED] flex items-center justify-center">
              <Brain className="w-6 h-6 text-black" />
            </div>
            <span className="text-xl font-bold font-['Outfit']">NeuroSense AI</span>
          </div>
          <Button 
            data-testid="nav-get-started-btn"
            onClick={handleGetStarted}
            className="bg-[#00F0FF] text-black font-bold hover:shadow-[0_0_30px_rgba(0,240,255,0.5)] transition-all"
          >
            {user ? 'Dashboard' : 'Get Started'}
          </Button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 relative overflow-hidden">
        <div className="absolute inset-0 opacity-30">
          <img 
            src="https://images.unsplash.com/photo-1674027215016-0a4abfdbf1cc?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzV8MHwxfHNlYXJjaHwyfHxmdXR1cmlzdGljJTIwYnJhaW4lMjBzY2FuJTIwaG9sb2dyYW0lMjBkYXRhfGVufDB8fHx8MTc2NDU5MzQ4NXww&ixlib=rb-4.1.0&q=85"
            alt="Brain hologram"
            className="w-full h-full object-cover"
          />
        </div>
        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="inline-block mb-6 px-6 py-2 rounded-full border border-[#00F0FF]/30 bg-black/40 backdrop-blur-xl">
              <span className="text-xs font-medium tracking-widest uppercase text-[#00F0FF]">Multimodal Intelligence System</span>
            </div>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6 tracking-tight font-['Outfit']">
              The Future of
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-[#00F0FF] to-[#7C3AED] mt-2">
                Brain Health Monitoring
              </span>
            </h1>
            <p className="text-lg sm:text-xl text-white/70 mb-10 max-w-2xl mx-auto leading-relaxed">
              Fusing vocal patterns, movement dynamics, and social interactions through advanced AI to revolutionize TBI research and cognitive health monitoring.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                data-testid="hero-get-started-btn"
                onClick={handleGetStarted}
                className="text-lg px-8 py-6 bg-[#00F0FF] text-black font-bold hover:shadow-[0_0_40px_rgba(0,240,255,0.6)] transition-all rounded-full"
              >
                Get Started
              </Button>
              <Button 
                data-testid="hero-learn-more-btn"
                onClick={() => document.getElementById('features').scrollIntoView({ behavior: 'smooth' })}
                variant="outline"
                className="text-lg px-8 py-6 border-white/20 hover:bg-white/5 rounded-full"
              >
                Learn More
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl sm:text-5xl font-bold mb-4 font-['Outfit']">Multimodal Intelligence</h2>
            <p className="text-white/60 text-lg max-w-2xl mx-auto">
              Advanced AI algorithms analyzing multiple data streams for comprehensive brain health insights
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-card p-8 rounded-xl hover:border-[#00F0FF]/30 transition-all duration-300 group"
                >
                  <div className="w-14 h-14 rounded-lg bg-gradient-to-br from-[#00F0FF]/20 to-[#7C3AED]/20 flex items-center justify-center mb-6 group-hover:shadow-[0_0_30px_rgba(0,240,255,0.3)] transition-all">
                    <Icon className="w-7 h-7 text-[#00F0FF]" />
                  </div>
                  <h3 className="text-xl font-bold mb-3 font-['Outfit']">{feature.title}</h3>
                  <p className="text-white/60 leading-relaxed">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Research Section */}
      <section className="py-20 px-6 bg-gradient-to-b from-transparent to-[#0B0F19]">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <div className="inline-block mb-4 px-4 py-1 rounded-full border border-[#7C3AED]/30 bg-black/40">
                <span className="text-xs font-medium tracking-widest uppercase text-[#7C3AED]">For Researchers</span>
              </div>
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 font-['Outfit']">Advanced Research Tools</h2>
              <p className="text-white/70 text-lg mb-6 leading-relaxed">
                Comprehensive data collection and analysis tools for TBI research. Export raw data, visualize trends across patient cohorts, and generate clinical-grade reports.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-[#7C3AED]/20 flex items-center justify-center flex-shrink-0 mt-1">
                    <div className="w-2 h-2 rounded-full bg-[#7C3AED]"></div>
                  </div>
                  <span className="text-white/80">Multi-patient data aggregation and comparison</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-[#7C3AED]/20 flex items-center justify-center flex-shrink-0 mt-1">
                    <div className="w-2 h-2 rounded-full bg-[#7C3AED]"></div>
                  </div>
                  <span className="text-white/80">CSV and JSON data export capabilities</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-[#7C3AED]/20 flex items-center justify-center flex-shrink-0 mt-1">
                    <div className="w-2 h-2 rounded-full bg-[#7C3AED]"></div>
                  </div>
                  <span className="text-white/80">Real-time statistics and trend analysis</span>
                </li>
              </ul>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              <img 
                src="https://images.unsplash.com/photo-1581595220921-eec2071e5159?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1ODF8MHwxfHNlYXJjaHwzfHxtZWRpY2FsJTIwcmVzZWFyY2hlciUyMGRhcmslMjBsYWIlMjBmdXR1cmlzdGljJTIwc2NyZWVuc3xlbnwwfHx8fDE3NjQ1OTM0ODd8MA&ixlib=rb-4.1.0&q=85"
                alt="Research lab"
                className="rounded-2xl shadow-2xl"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center glass-card p-12 rounded-2xl"
        >
          <h2 className="text-4xl sm:text-5xl font-bold mb-6 font-['Outfit']">Ready to Get Started?</h2>
          <p className="text-white/70 text-lg mb-8">
            Join the future of cognitive health monitoring with NeuroSense AI
          </p>
          <Button 
            data-testid="cta-get-started-btn"
            onClick={handleGetStarted}
            className="text-lg px-10 py-6 bg-[#00F0FF] text-black font-bold hover:shadow-[0_0_40px_rgba(0,240,255,0.6)] transition-all rounded-full"
          >
            Get Started Now
          </Button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-8 px-6">
        <div className="max-w-7xl mx-auto text-center text-white/50 text-sm">
          <p>© 2025 NeuroSense AI. Advancing brain health through multimodal intelligence.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
