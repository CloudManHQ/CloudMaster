import { NavLink } from "react-router-dom";
import { LayoutDashboard, BarChart3, FileText, Settings, BookOpen } from "lucide-react";
import { cn } from "@/utils/cn";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: BarChart3, label: "Analytics", href: "/analytics" },
  { icon: FileText, label: "Content", href: "/content" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

export function Sidebar() {
  return (
    <aside className="w-64 border-r bg-card">
      <div className="flex h-14 items-center border-b px-6">
        <BookOpen className="mr-2 h-5 w-5" />
        <span className="font-semibold">AI Guru Console</span>
      </div>
      <nav className="space-y-1 p-4">
        {navItems.map((item) => (
          <NavLink
            key={item.href}
            to={item.href}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )
            }
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
