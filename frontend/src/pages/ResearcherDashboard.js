import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  fetchPatients,
  fetchStatistics,
} from "../api"; // USE API WRAPPER
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  Brain,
  Users,
  Activity,
  Download,
  LogOut,
  TrendingUp,
  AlertCircle,
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

const ResearcherDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [patients, setPatients] = useState([]);
  const [statistics, setStatistics] = useState(null);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    try {
      const [ptsRes, statsRes] = await Promise.all([
        fetchPatients(),
        fetchStatistics(),
      ]);

      setPatients(ptsRes.data || []);
      setStatistics(statsRes.data || {});
    } catch (err) {
      console.error("Dashboard load error:", err);
    }
    setLoading(false);
  };

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#030712] flex items-center justify-center text-white/60">
        Loading researcher dashboard...
      </div>
    );
  }

  const scoreData = statistics
    ? [
        { name: "Overall", value: Math.round(statistics.average_scores.overall) },
        { name: "Vocal", value: Math.round(statistics.average_scores.vocal) },
        { name: "Movement", value: Math.round(statistics.average_scores.movement) },
        { name: "Social", value: Math.round(statistics.average_scores.social) },
      ]
    : [];

  const COLORS = ["#00F0FF", "#7C3AED", "#10B981", "#F59E0B"];

  return (
    <div className="min-h-screen bg-[#030712] text-white">
      <nav className="border-b border-white/10 sticky top-0 backdrop-blur-xl z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00F0FF] to-[#7C3AED] flex items-center justify-center">
              <Brain className="w-6 h-6 text-black" />
            </div>
            <div>
              <span className="text-xl font-bold">NeuroSense AI</span>
              <p className="text-xs text-white/50">Research Dashboard</p>
            </div>
          </div>

          <Button variant="ghost" onClick={handleLogout} className="text-white/60">
            <LogOut className="w-5 h-5" />
          </Button>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* STATISTICS CARDS */}
        {statistics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <CardHeader>
                <CardDescription>Total Patients</CardDescription>
                <CardTitle>{statistics.total_patients}</CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>Sensor Readings</CardDescription>
                <CardTitle>{statistics.total_sensor_readings}</CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>Total Alerts</CardDescription>
                <CardTitle>{statistics.total_alerts}</CardTitle>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>Avg Score</CardDescription>
                <CardTitle>
                  {Math.round(statistics.average_scores.overall)}
                </CardTitle>
              </CardHeader>
            </Card>
          </div>
        )}

        {/* SCORE BARCHART */}
        <Card>
          <CardHeader>
            <CardTitle>Average Scores</CardTitle>
            <CardDescription>Category-wise performance</CardDescription>
          </CardHeader>
          <CardContent>
            {scoreData.length > 0 && (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={scoreData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="value">
                    {scoreData.map((entry, index) => (
                      <Cell key={index} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* PATIENT TABLE */}
        <Card>
          <CardHeader>
            <CardTitle>Patients</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="py-3">ID</th>
                  <th>Name</th>
                  <th>Email</th>
                  <th>Latest Score</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr
                    key={i}
                    className="border-b border-white/5 hover:bg-white/5 transition"
                  >
                    <td className="py-3">{p.id.slice(0, 8)}...</td>
                    <td>{p.name}</td>
                    <td>{p.email}</td>
                    <td>
                      {p.latest_metrics ? (
                        <span>{Math.round(p.latest_metrics.overall_score)}</span>
                      ) : (
                        <span className="text-white/50">N/A</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ResearcherDashboard;
