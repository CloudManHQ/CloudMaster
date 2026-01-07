export interface CloudVendor {
  id: string;
  name: string;
  logo: string;
  description: string;
  website: string;
}

export interface ProductCategory {
  id: string;
  name: string;
  description: string;
  parentId?: string;
  children?: ProductCategory[];
}

export interface ProductDocument {
  id: string;
  title: string;
  type: 'guide' | 'api' | 'faq' | 'tutorial' | 'whitepaper';
  url: string;
  lastUpdated: string;
}

export interface CloudProduct {
  id: string;
  name: string;
  description: string;
  categoryId: string;
  vendorId: string;
  documents: ProductDocument[];
  website: string;
  features: string[];
}

export interface VendorProducts {
  vendor: CloudVendor;
  categories: ProductCategory[];
  products: CloudProduct[];
}

export interface AppState {
  vendors: CloudVendor[];
  selectedVendorId: string | null;
  selectedCategoryId: string | null;
  selectedProductId: string | null;
  searchTerm: string;
}
