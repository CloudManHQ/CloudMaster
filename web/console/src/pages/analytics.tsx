import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Analytics</h1>
        <p className="text-muted-foreground">Detailed insights and metrics</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Traffic Overview</CardTitle>
            <CardDescription>Daily visitors over the past 30 days</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] flex items-center justify-center bg-muted rounded-md">
              <p className="text-muted-foreground">Chart placeholder</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Content Performance</CardTitle>
            <CardDescription>Top performing documentation</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] flex items-center justify-center bg-muted rounded-md">
              <p className="text-muted-foreground">Chart placeholder</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Search Analytics</CardTitle>
          <CardDescription>What users are searching for</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span>transformer</span>
              <div className="flex items-center gap-4">
                <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-[85%]" />
                </div>
                <span className="text-sm text-muted-foreground">1,234</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span>fine-tuning</span>
              <div className="flex items-center gap-4">
                <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-[65%]" />
                </div>
                <span className="text-sm text-muted-foreground">892</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span>ethics</span>
              <div className="flex items-center gap-4">
                <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-[45%]" />
                </div>
                <span className="text-sm text-muted-foreground">654</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
