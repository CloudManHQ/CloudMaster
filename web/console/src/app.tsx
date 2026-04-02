import { Routes, Route } from "react-router-dom";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import { DashboardPage } from "@/pages/dashboard";
import { AnalyticsPage } from "@/pages/analytics";
import { ContentPage } from "@/pages/content";
import { SettingsPage } from "@/pages/settings";

export function App() {
  return (
    <DashboardLayout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/content" element={<ContentPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </DashboardLayout>
  );
}
