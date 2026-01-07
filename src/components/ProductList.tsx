import React from 'react';
import { Card, Grid, Typography, Tag, Space, Button } from 'antd';
import type { CloudProduct } from '../types';
import { LinkOutlined, BookOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

interface ProductListProps {
  products: CloudProduct[];
  onProductSelect: (productId: string) => void;
}

const ProductList: React.FC<ProductListProps> = ({ products, onProductSelect }) => {
  const getDocumentTypeLabel = (type: string): string => {
    const typeMap: Record<string, string> = {
      guide: '指南',
      api: 'API文档',
      faq: '常见问题',
      tutorial: '教程',
      whitepaper: '白皮书'
    };
    return typeMap[type] || type;
  };

  return (
    <div className="product-list">
      <Title level={4}>产品列表</Title>
      <div style={{ display: 'flex', flexWrap: 'wrap', margin: '-8px' }}>
        {products.map(product => (
          <div key={product.id} style={{ padding: '8px', width: '100%' }}>
            <Grid xs={24} sm={12} md={8} lg={8} xl={6} xxl={6}>
              <Card
                title={product.name}
                extra={
                  <Space>
                    <Button
                      type="primary"
                      icon={<BookOutlined />}
                      onClick={() => onProductSelect(product.id)}
                      size="small"
                    >
                      查看文档
                    </Button>
                    <Button
                      icon={<LinkOutlined />}
                      href={product.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      size="small"
                    >
                      官方网站
                    </Button>
                  </Space>
                }
                hoverable
                style={{ 
                  borderRadius: 12, 
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                  transition: 'all 0.3s ease'
                }}
                bodyStyle={{ padding: 20 }}
              >
                <Text type="secondary">{product.description}</Text>
                <br />
                <br />
                <Space wrap>
                  {product.features.map((feature, index) => (
                    <Tag key={index} color="blue">{feature}</Tag>
                  ))}
                </Space>
                <br />
                <br />
                <Title level={5}>相关文档</Title>
              <div style={{ paddingLeft: 16 }}>
                {product.documents.map(doc => (
                  <div key={doc.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, padding: '4px 0' }}>
                    <div>
                      <div style={{ fontWeight: 500, marginBottom: 4 }}>{doc.title}</div>
                      <Space>
                        <Tag color="green" size="small">{getDocumentTypeLabel(doc.type)}</Tag>
                        <Text type="secondary" style={{ fontSize: 12 }}>最后更新: {doc.lastUpdated}</Text>
                      </Space>
                    </div>
                    <a href={doc.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 14 }}>
                      查看
                    </a>
                  </div>
                ))}
              </div>
              </Card>
            </Grid>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProductList;
