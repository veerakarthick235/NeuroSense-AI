import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { Brain, Activity, Users, TrendingUp, AlertCircle, Sparkles, LogOut, RefreshCw } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { toast } from 'sonner';
import { useAuth } from '../App';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const PatientDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [latestMetrics, setLatestMetrics] = useState(null);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [insights, setInsights] = useState([]);
  const [generatingInsight, setGeneratingInsight] = useState(false);
  const [simulatingData, setSimulatingData] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchLatestMetrics(),
        fetchMetricsHistory(),
        fetchAlerts(),
        fetchInsights()
      ]);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchLatestMetrics = async () => {
    try {
      const response = await axios.get(`${API}/metrics/latest`);
      setLatestMetrics(response.data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchMetricsHistory = async () => {
    try {
      const response = await axios.get(`${API}/metrics/history?days=7`);
      setMetricsHistory(response.data);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API}/alerts`);
      setAlerts(response.data);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const fetchInsights = async () => {
    try {
      const response = await axios.get(`${API}/insights`);
      setInsights(response.data);
    } catch (error) {
      console.error('Error fetching insights:', error);
    }
  };

  const handleSimulateData = async () => {
    setSimulatingData(true);
    try {
      await axios.post(`${API}/data/sensors/simulate`);
      await fetchLatestMetrics();
      await fetchMetricsHistory();
      toast.success('Simulated sensor data generated');
    } catch (error) {
      toast.error('Failed to simulate data');
    } finally {
      setSimulatingData(false);
    }
  };

  const handleGenerateInsight = async () => {
    setGeneratingInsight(true);
    try {
      const response = await axios.post(`${API}/insights/generate`, { user_id: user.id });
      setInsights([response.data, ...insights]);
      toast.success('AI insight generated');
    } catch (error) {
      toast.error('Failed to generate insight');
    } finally {
      setGeneratingInsight(false);
    }
  };

  const handleCheckAlerts = async () => {
    try {
      await axios.post(`${API}/alerts/check`);
      await fetchAlerts();
      toast.success('Alert check completed');
    } catch (error) {
      toast.error('Failed to check alerts');
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#030712] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-[#00F0FF] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white/60">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const radarData = latestMetrics ? [
    { subject: 'Vocal', value: latestMetrics.vocal_score, fullMark: 100 },
    { subject: 'Movement', value: latestMetrics.movement_score, fullMark: 100 },
    { subject: 'Social', value: latestMetrics.social_score, fullMark: 100 }
  ] : [];

  return (
    <div data-testid="patient-dashboard" className="min-h-screen bg-[#030712] text-white">
      {/* Header */}
      <nav className="glass backdrop-blur-xl border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00F0FF] to-[#7C3AED] flex items-center justify-center">
              <Brain className="w-6 h-6 text-black" />
            </div>
            <div>
              <span className="text-xl font-bold font-['Outfit']">NeuroSense AI</span>
              <p className="text-xs text-white/50">Patient Dashboard</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-white/60">Welcome, {user?.name}</span>
            <Button
              data-testid="logout-btn"
              onClick={handleLogout}
              variant="ghost"
              className="text-white/60 hover:text-white"
            >
              <LogOut className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Quick Actions */}
        <div className="flex flex-wrap gap-4">
          <Button
            data-testid="simulate-data-btn"
            onClick={handleSimulateData}
            disabled={simulatingData}
            className="bg-[#00F0FF] text-black font-bold hover:shadow-[0_0_30px_rgba(0,240,255,0.5)]"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${simulatingData ? 'animate-spin' : ''}`} />
            {simulatingData ? 'Generating...' : 'Simulate Data'}
          </Button>
          <Button
            data-testid="generate-insight-btn"
            onClick={handleGenerateInsight}
            disabled={generatingInsight}
            className="bg-[#7C3AED] text-white font-bold hover:shadow-[0_0_30px_rgba(124,58,237,0.5)]"
          >
            <Sparkles className={`w-4 h-4 mr-2 ${generatingInsight ? 'animate-spin' : ''}`} />
            {generatingInsight ? 'Generating...' : 'Generate AI Insight'}
          </Button>
        </div>

        {/* Health Score Overview */}
        {latestMetrics && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-4 gap-6"
          >
            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60">Overall Health</CardDescription>
                <CardTitle className="text-4xl font-bold font-['Outfit'] text-[#00F0FF]">
                  {latestMetrics.overall_score.toFixed(0)}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-white/50">Out of 100</div>
              </CardContent>
            </Card>

            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Vocal Patterns
                </CardDescription>
                <CardTitle className="text-3xl font-bold font-['Outfit']">
                  {latestMetrics.vocal_score.toFixed(0)}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Movement
                </CardDescription>
                <CardTitle className="text-3xl font-bold font-['Outfit']">
                  {latestMetrics.movement_score.toFixed(0)}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  Social
                </CardDescription>
                <CardTitle className="text-3xl font-bold font-['Outfit']">
                  {latestMetrics.social_score.toFixed(0)}
                </CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
        )}

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Trend Chart */}
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="font-['Outfit']">7-Day Trend</CardTitle>
              <CardDescription className="text-white/60">Overall health score over time</CardDescription>
            </CardHeader>
            <CardContent>
              {metricsHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={metricsHistory}>
                    <defs>
                      <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00F0FF" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#00F0FF" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis 
                      dataKey="timestamp" 
                      stroke="rgba(255,255,255,0.5)"
                      tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    />
                    <YAxis stroke="rgba(255,255,255,0.5)" domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ background: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      labelStyle={{ color: '#00F0FF' }}
                    />
                    <Area type="monotone" dataKey="overall_score" stroke="#00F0FF" strokeWidth={2} fillOpacity={1} fill="url(#colorScore)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[250px] flex items-center justify-center text-white/50">
                  No data available. Click "Simulate Data" to generate sample data.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Radar Chart */}
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="font-['Outfit']">Multimodal Analysis</CardTitle>
              <CardDescription className="text-white/60">Current health metrics breakdown</CardDescription>
            </CardHeader>
            <CardContent>
              {radarData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="subject" stroke="rgba(255,255,255,0.7)" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="rgba(255,255,255,0.3)" />
                    <Radar name="Score" dataKey="value" stroke="#00F0FF" fill="#00F0FF" fillOpacity={0.3} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[250px] flex items-center justify-center text-white/50">
                  No metrics available
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Alerts and Insights */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Alerts */}
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="font-['Outfit'] flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-[#F59E0B]" />
                Recent Alerts
              </CardTitle>
              <CardDescription className="text-white/60">TBI risk assessments and notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-[300px] overflow-y-auto">
                {alerts.length > 0 ? (
                  alerts.slice(0, 5).map((alert, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border ${
                        alert.severity === 'high' 
                          ? 'border-red-500/30 bg-red-500/10' 
                          : alert.severity === 'medium'
                          ? 'border-yellow-500/30 bg-yellow-500/10'
                          : 'border-blue-500/30 bg-blue-500/10'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className={`text-xs font-bold uppercase ${
                          alert.severity === 'high' ? 'text-red-400' : alert.severity === 'medium' ? 'text-yellow-400' : 'text-blue-400'
                        }`}>
                          {alert.severity} Risk
                        </span>
                        <span className="text-xs text-white/50">
                          {new Date(alert.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                      <p className="text-sm text-white/80">{alert.message}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-white/50">
                    No alerts. Your metrics are looking good!
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* AI Insights */}
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="font-['Outfit'] flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-[#7C3AED]" />
                AI Insights
              </CardTitle>
              <CardDescription className="text-white/60">Personalized health recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-[300px] overflow-y-auto">
                {insights.length > 0 ? (
                  insights.slice(0, 3).map((insight, index) => (
                    <div
                      key={index}
                      className="p-4 rounded-lg border border-[#7C3AED]/30 bg-[#7C3AED]/10"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-xs font-bold uppercase text-[#7C3AED]">
                          {insight.insight_type.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-white/50">
                          {new Date(insight.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                      <p className="text-sm text-white/80 leading-relaxed whitespace-pre-wrap">{insight.content}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-white/50">
                    No insights yet. Click "Generate AI Insight" to get personalized recommendations.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default PatientDashboard;