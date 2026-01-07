
import { render, screen, fireEvent } from '@testing-library/react';
import CategoryNav from './CategoryNav';
import { ProductCategory } from '../types';

const mockCategories: ProductCategory[] = [
  {
    id: 'compute',
    name: '计算',
    description: '提供各种计算资源',
    children: [
      {
        id: 'compute-vm',
        name: '虚拟机',
        description: '弹性云服务器、裸金属服务器等',
        parentId: 'compute'
      },
      {
        id: 'compute-container',
        name: '容器服务',
        description: '容器编排、容器镜像服务等',
        parentId: 'compute'
      }
    ]
  },
  {
    id: 'storage',
    name: '存储',
    description: '提供各种存储服务',
    children: [
      {
        id: 'storage-object',
        name: '对象存储',
        description: '海量、安全、低成本的云存储服务',
        parentId: 'storage'
      }
    ]
  }
];

describe('CategoryNav Component', () => {
  test('should render category nav with categories', () => {
    render(
      <CategoryNav
        categories={mockCategories}
        selectedCategoryId={null}
        onCategoryChange={() => {}}
      />
    );
    
    expect(screen.getByText('产品类别')).toBeInTheDocument();
    expect(screen.getByText('计算')).toBeInTheDocument();
    expect(screen.getByText('存储')).toBeInTheDocument();
  });

  test('should expand and collapse categories', () => {
    render(
      <CategoryNav
        categories={mockCategories}
        selectedCategoryId={null}
        onCategoryChange={() => {}}
      />
    );
    
    // 默认应该是展开的
    expect(screen.getByText('虚拟机')).toBeInTheDocument();
    expect(screen.getByText('容器服务')).toBeInTheDocument();
    expect(screen.getByText('对象存储')).toBeInTheDocument();
  });

  test('should call onCategoryChange when category is selected', () => {
    const mockOnCategoryChange = jest.fn();
    
    render(
      <CategoryNav
        categories={mockCategories}
        selectedCategoryId={null}
        onCategoryChange={mockOnCategoryChange}
      />
    );
    
    fireEvent.click(screen.getByText('虚拟机'));
    expect(mockOnCategoryChange).toHaveBeenCalledWith('compute-vm');
    
    fireEvent.click(screen.getByText('对象存储'));
    expect(mockOnCategoryChange).toHaveBeenCalledWith('storage-object');
  });

  test('should display selected category correctly', () => {
    render(
      <CategoryNav
        categories={mockCategories}
        selectedCategoryId='compute-vm'
        onCategoryChange={() => {}}
      />
    );
    
    // 在Ant Design的Tree组件中，选中的节点会有ant-tree-node-selected类
    const vmNode = screen.getByText('虚拟机').closest('.ant-tree-node');
    expect(vmNode).toHaveClass('ant-tree-node-selected');
    
    const containerNode = screen.getByText('容器服务').closest('.ant-tree-node');
    expect(containerNode).not.toHaveClass('ant-tree-node-selected');
  });
});
