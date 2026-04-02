import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, Users, Eye, TrendingUp } from "lucide-react";

const stats = [
  { title: "Total Documents", value: "290", icon: FileText, change: "+12%" },
  { title: "Active Users", value: "1,234", icon: Users, change: "+5%" },
  { title: "Page Views", value: "45.2K", icon: Eye, change: "+18%" },
  { title: "Growth Rate", value: "23%", icon: TrendingUp, change: "+2%" },
];

export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">Overview of your knowledge base</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                <span className="text-green-600">{stat.change}</span> from last month
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest updates to the knowledge base</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              <li className="flex items-center justify-between">
                <span className="text-sm">Updated AI Fundamentals</span>
                <span className="text-xs text-muted-foreground">2 hours ago</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-sm">Added new case study</span>
                <span className="text-xs text-muted-foreground">5 hours ago</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-sm">Published ethics guide</span>
                <span className="text-xs text-muted-foreground">1 day ago</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Popular Content</CardTitle>
            <CardDescription>Most viewed documentation</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              <li className="flex items-center justify-between">
                <span className="text-sm">ChatGPT Guide</span>
                <span className="text-xs text-muted-foreground">2.4K views</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-sm">Transformer Architecture</span>
                <span className="text-xs text-muted-foreground">1.8K views</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-sm">AI Ethics</span>
                <span className="text-xs text-muted-foreground">1.2K views</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
