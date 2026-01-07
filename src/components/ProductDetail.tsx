import React from 'react';
import { Card, Typography, Tag, Space, Button, Divider, Descriptions } from 'antd';
import type { CloudProduct } from '../types';
import { LinkOutlined, ArrowLeftOutlined, CalendarOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

interface ProductDetailProps {
  product: CloudProduct | undefined;
  onBack: () => void;
}

const ProductDetail: React.FC<ProductDetailProps> = ({ product, onBack }) => {
  if (!product) {
    return (
      <div className="product-detail">
        <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
          返回列表
        </Button>
        <br />
        <br />
        <Card>
          <Title level={4}>产品不存在</Title>
        </Card>
      </div>
    );
  }

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

  const getDocumentTypeColor = (type: string): string => {
    const colorMap: Record<string, string> = {
      guide: 'blue',
      api: 'green',
      faq: 'orange',
      tutorial: 'purple',
      whitepaper: 'red'
    };
    return colorMap[type] || 'default';
  };

  return (
    <div className="product-detail">
      <Button icon={<ArrowLeftOutlined />} onClick={onBack} style={{ marginBottom: 16 }}>
        返回产品列表
      </Button>
      
      <Card
        title={
          <Space>
            {product.name}
            <Tag color="blue">{product.vendorId.toUpperCase()}</Tag>
          </Space>
        }
        extra={
          <Button
            icon={<LinkOutlined />}
            href={product.website}
            target="_blank"
            rel="noopener noreferrer"
          >
            访问产品官网
          </Button>
        }
        bordered={false}
      >
        <div style={{ marginBottom: 24 }}>
          <Text type="secondary" style={{ fontSize: '16px', lineHeight: '1.6' }}>
            {product.description}
          </Text>
        </div>

        <Descriptions bordered column={2} size="middle">
          <Descriptions.Item label="所属类别">{product.categoryId ? product.categoryId.split('-').join(' ') : '未分类'}</Descriptions.Item>
          <Descriptions.Item label="产品官网">
            <a href={product.website} target="_blank" rel="noopener noreferrer">
              {product.website} <LinkOutlined style={{ fontSize: 12 }} />
            </a>
          </Descriptions.Item>
        </Descriptions>

        <Divider>
          <Title level={5} style={{ margin: 0 }}>产品特性</Title>
        </Divider>
        
        <Space wrap style={{ marginBottom: 24 }}>
          {product.features.map((feature, index) => (
            <Tag key={index} color="blue">{feature}</Tag>
          ))}
        </Space>

        <Divider>
          <Title level={5} style={{ margin: 0 }}>相关文档</Title>
        </Divider>
        
        <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 16 }}>
          {product.documents.map(doc => (
            <div key={doc.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, padding: '8px 0', borderBottom: '1px solid #f5f5f5' }}>
              <div>
                <div style={{ fontWeight: 500, marginBottom: 8 }}>
                  <Space size="middle">
                    {doc.title}
                    <Tag color={getDocumentTypeColor(doc.type)}>
                      {getDocumentTypeLabel(doc.type)}
                    </Tag>
                  </Space>
                </div>
                <Space>
                  <CalendarOutlined style={{ fontSize: 12, color: '#8c8c8c' }} />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    最后更新: {doc.lastUpdated}
                  </Text>
                </Space>
              </div>
              <a href={doc.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 14 }}>
                查看文档 <LinkOutlined style={{ fontSize: 12 }} />
              </a>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default ProductDetail;
