import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Brain, Users, Activity, Download, LogOut, TrendingUp, AlertCircle } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { toast } from 'sonner';
import { useAuth } from '../App';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ResearcherDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [patients, setPatients] = useState([]);
  const [statistics, setStatistics] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchPatients(),
        fetchStatistics()
      ]);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPatients = async () => {
    try {
      const response = await axios.get(`${API}/research/patients`);
      setPatients(response.data);
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await axios.get(`${API}/research/statistics`);
      setStatistics(response.data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    }
  };

  const handleExportData = async () => {
    try {
      const response = await axios.get(`${API}/export/data?format=json`);
      const dataStr = JSON.stringify(response.data, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `neurosense-export-${new Date().toISOString()}.json`;
      link.click();
      toast.success('Data exported successfully');
    } catch (error) {
      toast.error('Failed to export data');
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
          <p className="text-white/60">Loading research dashboard...</p>
        </div>
      </div>
    );
  }

  const scoreDistributionData = statistics ? [
    { name: 'Overall', value: Math.round(statistics.average_scores.overall) },
    { name: 'Vocal', value: Math.round(statistics.average_scores.vocal) },
    { name: 'Movement', value: Math.round(statistics.average_scores.movement) },
    { name: 'Social', value: Math.round(statistics.average_scores.social) }
  ] : [];

  const COLORS = ['#00F0FF', '#7C3AED', '#10B981', '#F59E0B'];

  return (
    <div data-testid="researcher-dashboard" className="min-h-screen bg-[#030712] text-white">
      {/* Header */}
      <nav className="glass backdrop-blur-xl border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00F0FF] to-[#7C3AED] flex items-center justify-center">
              <Brain className="w-6 h-6 text-black" />
            </div>
            <div>
              <span className="text-xl font-bold font-['Outfit']">NeuroSense AI</span>
              <p className="text-xs text-white/50">Research Dashboard</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Button
              data-testid="export-data-btn"
              onClick={handleExportData}
              className="bg-[#7C3AED] text-white font-bold hover:shadow-[0_0_30px_rgba(124,58,237,0.5)]"
            >
              <Download className="w-4 h-4 mr-2" />
              Export Data
            </Button>
            <span className="text-sm text-white/60">Welcome, {user?.name}</span>
            <Button
              data-testid="researcher-logout-btn"
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
        {/* Statistics Overview */}
        {statistics && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-4 gap-6"
          >
            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  Total Patients
                </CardDescription>
                <CardTitle className="text-4xl font-bold font-['Outfit'] text-[#00F0FF]">
                  {statistics.total_patients}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Sensor Readings
                </CardDescription>
                <CardTitle className="text-4xl font-bold font-['Outfit'] text-[#7C3AED]">
                  {statistics.total_sensor_readings.toLocaleString()}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Total Alerts
                </CardDescription>
                <CardTitle className="text-4xl font-bold font-['Outfit'] text-[#F59E0B]">
                  {statistics.total_alerts}
                </CardTitle>
              </CardHeader>
            </Card>

            <Card className="glass-card border-white/10">
              <CardHeader className="pb-3">
                <CardDescription className="text-white/60 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Avg Health Score
                </CardDescription>
                <CardTitle className="text-4xl font-bold font-['Outfit'] text-[#10B981]">
                  {Math.round(statistics.average_scores.overall)}
                </CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
        )}

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Bar Chart */}
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="font-['Outfit']">Average Scores by Category</CardTitle>
              <CardDescription className="text-white/60">Population-level metrics analysis</CardDescription>
            </CardHeader>
            <CardContent>
              {scoreDistributionData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={scoreDistributionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" />
                    <YAxis stroke="rgba(255,255,255,0.5)" domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ background: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      labelStyle={{ color: '#00F0FF' }}
                    />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {scoreDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-white/50">
                  No data available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Pie Chart */}
          <Card className="glass-card border-white/10">
            <CardHeader>
              <CardTitle className="font-['Outfit']">Score Distribution</CardTitle>
              <CardDescription className="text-white/60">Comparative analysis</CardDescription>
            </CardHeader>
            <CardContent>
              {scoreDistributionData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={scoreDistributionData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {scoreDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ background: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-white/50">
                  No data available
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Patient List */}
        <Card className="glass-card border-white/10">
          <CardHeader>
            <CardTitle className="font-['Outfit']">Patient Cohort</CardTitle>
            <CardDescription className="text-white/60">Overview of all registered patients</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 font-['JetBrains_Mono'] text-sm text-white/60">Patient ID</th>
                    <th className="text-left py-3 px-4 font-['JetBrains_Mono'] text-sm text-white/60">Name</th>
                    <th className="text-left py-3 px-4 font-['JetBrains_Mono'] text-sm text-white/60">Email</th>
                    <th className="text-left py-3 px-4 font-['JetBrains_Mono'] text-sm text-white/60">Latest Score</th>
                    <th className="text-left py-3 px-4 font-['JetBrains_Mono'] text-sm text-white/60">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.length > 0 ? (
                    patients.map((patient, index) => (
                      <tr key={index} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                        <td className="py-3 px-4 font-['JetBrains_Mono'] text-sm text-white/80">{patient.id.slice(0, 8)}...</td>
                        <td className="py-3 px-4 text-white">{patient.name}</td>
                        <td className="py-3 px-4 text-white/60">{patient.email}</td>
                        <td className="py-3 px-4">
                          {patient.latest_metrics ? (
                            <span className="font-bold text-[#00F0FF]">
                              {Math.round(patient.latest_metrics.overall_score)}
                            </span>
                          ) : (
                            <span className="text-white/40">N/A</span>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          {patient.latest_metrics ? (
                            patient.latest_metrics.overall_score >= 75 ? (
                              <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-xs font-bold">Good</span>
                            ) : patient.latest_metrics.overall_score >= 60 ? (
                              <span className="px-3 py-1 rounded-full bg-yellow-500/20 text-yellow-400 text-xs font-bold">Monitor</span>
                            ) : (
                              <span className="px-3 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-bold">Alert</span>
                            )
                          ) : (
                            <span className="px-3 py-1 rounded-full bg-white/10 text-white/40 text-xs font-bold">No Data</span>
                          )}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="5" className="text-center py-12 text-white/50">
                        No patients registered yet
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ResearcherDashboard;
