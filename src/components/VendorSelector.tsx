import React from 'react';
import { Radio, Space, Typography } from 'antd';
import type { CloudVendor } from '../types';

const { Title } = Typography;

interface VendorSelectorProps {
  vendors: CloudVendor[];
  selectedVendorId: string | null;
  onVendorChange: (vendorId: string) => void;
}

const VendorSelector: React.FC<VendorSelectorProps> = ({
  vendors,
  selectedVendorId,
  onVendorChange
}) => {
  return (
    <div className="vendor-selector">
      <Title level={4}>选择云厂商</Title>
      <Radio.Group
        value={selectedVendorId}
        onChange={(e) => onVendorChange(e.target.value)}
        buttonStyle="solid"
        size="middle"
      >
        <Space direction="vertical" wrap>
          {vendors.map(vendor => (
            <Radio.Button key={vendor.id} value={vendor.id}>
              {vendor.name}
            </Radio.Button>
          ))}
        </Space>
      </Radio.Group>
    </div>
  );
};

export default VendorSelector;
