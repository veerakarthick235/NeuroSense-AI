import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api"; // FIXED
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import {
  Brain,
  Activity,
  Users,
  TrendingUp,
  AlertCircle,
  Sparkles,
  LogOut,
  RefreshCw,
} from "lucide-react";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { toast } from "sonner";
import { useAuth } from "../App";

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
        fetchInsights(),
      ]);
    } catch (err) {
      console.error("Error loading dashboard:", err);
    }
    setLoading(false);
  };

  const fetchLatestMetrics = async () => {
    const res = await api.get("/metrics/latest");
    setLatestMetrics(res.data);
  };

  const fetchMetricsHistory = async () => {
    const res = await api.get("/metrics/history?days=7");
    setMetricsHistory(res.data);
  };

  const fetchAlerts = async () => {
    const res = await api.post("/alerts/check"); // FIXED
    setAlerts(res.data.alerts || []);
  };

  const fetchInsights = async () => {
    // Insights list is optional in backend — avoid crash
    setInsights([]);
  };

  const handleSimulateData = async () => {
    setSimulatingData(true);
    try {
      await api.post("/data/sensors/simulate");
      await fetchLatestMetrics();
      await fetchMetricsHistory();
      toast.success("Simulated sensor data generated.");
    } catch {
      toast.error("Failed to simulate data.");
    }
    setSimulatingData(false);
  };

  const handleGenerateInsight = async () => {
    setGeneratingInsight(true);
    try {
      const res = await api.post("/insights/generate", { user_id: user.id });
      setInsights([res.data, ...insights]);
      toast.success("AI Insight generated");
    } catch {
      toast.error("Failed to generate insight.");
    }
    setGeneratingInsight(false);
  };

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#030712] flex items-center justify-center">
        <p className="text-white/60">Loading dashboard...</p>
      </div>
    );
  }

  const radarData = latestMetrics
    ? [
        { subject: "Vocal", value: latestMetrics.vocal_score, fullMark: 100 },
        { subject: "Movement", value: latestMetrics.movement_score, fullMark: 100 },
        { subject: "Social", value: latestMetrics.social_score, fullMark: 100 },
      ]
    : [];

  return (
    <div className="min-h-screen bg-[#030712] text-white">
      {/* HEADER */}
      <nav className="glass border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00F0FF] to-[#7C3AED] flex items-center justify-center">
              <Brain className="w-6 h-6 text-black" />
            </div>
            <div>
              <span className="text-xl font-bold">NeuroSense AI</span>
              <p className="text-xs text-white/50">Patient Dashboard</p>
            </div>
          </div>

          <Button onClick={handleLogout} variant="ghost" className="text-white/60">
            <LogOut className="w-5 h-5" />
          </Button>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* ACTION BUTTONS */}
        <div className="flex gap-4">
          <Button
            onClick={handleSimulateData}
            disabled={simulatingData}
            className="bg-[#00F0FF] text-black"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${simulatingData ? "animate-spin" : ""}`} />
            {simulatingData ? "Generating..." : "Simulate Data"}
          </Button>

          <Button
            onClick={handleGenerateInsight}
            disabled={generatingInsight}
            className="bg-[#7C3AED] text-white"
          >
            <Sparkles className={`w-4 h-4 mr-2 ${generatingInsight ? "animate-spin" : ""}`} />
            {generatingInsight ? "Generating..." : "Generate Insight"}
          </Button>
        </div>

        {/* METRICS */}
        {latestMetrics && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-4 gap-6"
          >
            <Card className="glass-card">
              <CardHeader>
                <CardDescription>Overall Health</CardDescription>
                <CardTitle>{latestMetrics.overall_score.toFixed(0)}</CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card">
              <CardHeader>
                <CardDescription>Vocal</CardDescription>
                <CardTitle>{latestMetrics.vocal_score.toFixed(0)}</CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card">
              <CardHeader>
                <CardDescription>Movement</CardDescription>
                <CardTitle>{latestMetrics.movement_score.toFixed(0)}</CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card">
              <CardHeader>
                <CardDescription>Social</CardDescription>
                <CardTitle>{latestMetrics.social_score.toFixed(0)}</CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
        )}

        {/* CHARTS */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>7-Day Trend</CardTitle>
            </CardHeader>
            <CardContent>
              {metricsHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={metricsHistory}>
                    <defs>
                      <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00F0FF" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#00F0FF" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="white" opacity={0.1} />
                    <XAxis dataKey="timestamp" stroke="white" opacity={0.7} />
                    <YAxis stroke="white" opacity={0.7} domain={[0, 100]} />
                    <Tooltip />
                    <Area dataKey="overall_score" stroke="#00F0FF" fill="url(#grad)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-white/50">No history yet.</p>
              )}
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Multimodal Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              {radarData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar dataKey="value" stroke="#00F0FF" fill="#00F0FF" fillOpacity={0.4} />
                  </RadarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-white/50">No metrics available</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ALERTS */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length > 0 ? (
              alerts.map((a, i) => (
                <div key={i} className="p-4 border rounded mb-3">
                  <p className="text-sm font-bold">{a.severity.toUpperCase()}</p>
                  <p>{a.message}</p>
                </div>
              ))
            ) : (
              <p className="text-white/50">No alerts.</p>
            )}
          </CardContent>
        </Card>

        {/* INSIGHTS */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>AI Insights</CardTitle>
          </CardHeader>
          <CardContent>
            {insights.length > 0 ? (
              insights.map((i, idx) => (
                <div key={idx} className="p-4 border rounded mb-3">
                  <p className="text-xs text-white/50">
                    {new Date(i.timestamp).toLocaleString()}
                  </p>
                  <p>{i.content}</p>
                </div>
              ))
            ) : (
              <p className="text-white/50">No insights yet.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PatientDashboard;
