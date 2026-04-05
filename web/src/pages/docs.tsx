import { Link } from "react-router-dom";
import { Folder, FileText } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { docSections } from "@/data/docMap";

export function DocsPage() {
  return (
    <div className="container py-8">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold mb-6">Documentation</h1>
        <p className="text-muted-foreground mb-8">
          Browse our comprehensive AI knowledge base organized by topics.
        </p>

        <div className="space-y-6">
          {docSections.map((section) => (
            <Card key={section.id}>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Folder className="h-5 w-5 text-primary" />
                  <CardTitle>{section.title}</CardTitle>
                </div>
                <p className="text-sm text-muted-foreground">
                  {section.description}
                </p>
              </CardHeader>
              <CardContent>
                <ul className="grid gap-2 sm:grid-cols-2">
                  {section.docs.map((doc) => (
                    <li key={doc.slug}>
                      <Link
                        to={`/docs/${section.id}-${doc.slug}`}
                        className="flex items-center gap-2 p-2 rounded-md hover:bg-muted transition-colors"
                      >
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{doc.title}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
