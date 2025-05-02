import argparse

import numpy
import torch
import onnx_tool


def parse_args():
    parser = argparse.ArgumentParser(description="get flops of onnx model")
    # parser.add_argument("config", help="config file path")
    parser.add_argument("path", help="onnx file path (folder) e.g: /root/autodl-tmp/model/swint/")
    parser.add_argument("export_path", help="export file path e.g: /root/autodl-tmp/results/")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # TODO：LiDAR骨干稀疏卷积及相关模块适配

    print("==================Start registering SparseConvolution==================")

    @onnx_tool.NODE_REGISTRY.register()
    class SparseConvolutionNode(onnx_tool.Node):
        def __init__(self, nodeproto):
            super().__init__(nodeproto)
            # 添加必要的默认属性
            self.add_default_value('kernel_size', [3, 3, 3])
            self.add_default_value('stride', [1, 1, 1])
            self.add_default_value('padding', [0, 0, 0])
            self.add_default_value('dilation', [1, 1, 1])
            self.add_default_value('groups', 1)
            self.add_default_value('output_channels', None)
            self.total_ops = 0  # 用于记录MACs

        def shape_infer(self, intensors: [], outtensors: []):
            """形状推理：计算输出张量的形状"""
            # if len(intensors) < 3:
            #     print(f"SparseConvolution缺少必要的输入: {self.name}")
            #     return
            input_shape = intensors[2].get_numpy().tolist()  # 输入张量的完整形状

            # 尝试获取权重，如果没有则使用默认值或从属性获取
            if len(intensors) > 3:
                weight = intensors[3].get_numpy()  # 卷积核权重
                output_channels = weight.shape[0]
            else:
                # 从属性获取或使用默认值
                output_channels = 256
                # print(f"SparseConvolution({self.name})缺少权重输入，使用默认输出通道数: {output_channels}")

            # 获取空间维度数（排除batch和channel维度）
            spatial_dims = len(input_shape) - 2

            # 确保所有参数列表长度与空间维度数匹配
            def adjust_param(param, name):
                if isinstance(param, int):
                    # 转换为列表
                    param = [param] * spatial_dims
                elif len(param) < spatial_dims:
                    # 扩展列表以匹配空间维度
                    param = list(param) + [param[-1]] * (spatial_dims - len(param))
                    # print(f"SparseConvolution({self.name})的{name}参数长度不足，扩展为: {param}")
                return param

            # 调整各参数列表
            padding = adjust_param(self.padding, "padding")
            kernel_size = adjust_param(self.kernel_size, "kernel_size")
            stride = adjust_param(self.stride, "stride")
            dilation = adjust_param(self.dilation, "dilation")

            # 计算空间维度的输出大小
            output_spatial_shape = []
            for i in range(spatial_dims):
                # 注意：input_shape[i+2] 跳过batch和channel维度
                dim_size = (input_shape[i + 2] + 2 * padding[i] -
                            dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
                output_spatial_shape.append(dim_size)

            # 组合输出形状：[batch_size, output_channels, spatial_dims...]
            output_shape = [input_shape[0], output_channels] + output_spatial_shape

            # 更新输出张量的形状和数据类型
            outtensors[0].update_shape(output_shape)
            outtensors[0].update_dtype(intensors[1].dtype)  # values的数据类型

        def value_infer(self, intensors: [], outtensors: []):
            return
            """值推理：估算输出张量的值和计算量"""
            # if len(intensors) < 3:
            #     print(f"SparseConvolution缺少必要的输入: {self.name}")
            #     return

            # try:
            #     indices = intensors[0].get_numpy()  # 稀疏索引
            #     values = intensors[1].get_numpy()  # 稀疏值
            #     input_shape = intensors[2].get_numpy()  # 输入张量的完整形状
            #
            #     # 尝试获取权重
            #     if len(intensors) > 3:
            #         weight = intensors[3].get_numpy()  # 卷积核权重
            #     else:
            #         # 创建默认权重（仅用于形状推理，不用于实际计算）
            #         in_channels = values.shape[1]
            #         output_channels = 256
            #         kernel_size = self.kernel_size if hasattr(self, 'kernel_size') else [3, 3, 3]
            #         weight_shape = [output_channels, in_channels // self.groups, *kernel_size]
            #         weight = numpy.random.randn(*weight_shape).astype(values.dtype)
            #         print(f"SparseConvolution({self.name})缺少权重输入，使用随机权重进行形状推理")
            #
            #     # 计算稀疏卷积的MACs
            #     self._compute_sparse_conv_macs(indices, values, input_shape, weight)
            #
            #     # 简化处理：无法计算实际输出值，只设置形状
            #     self.shape_infer(intensors, outtensors)
            #
            # except Exception as e:
            #     print(f"无法执行SparseConvolution的值推理: {e}")
            #     # 无法计算实际值时，至少设置形状
            #     self.shape_infer(intensors, outtensors)

        def _compute_sparse_conv_macs(self, indices, values, input_shape, weight):
            """计算稀疏卷积的MACs"""
            # 输入通道数和输出通道数
            in_channels = values.shape[1]
            out_channels = weight.shape[0] if weight is not None else 256

            # 计算每个卷积核的元素数量
            kernel_size = self.kernel_size if hasattr(self, 'kernel_size') else [3, 3, 3]
            kernel_elements = 1
            for k in kernel_size:
                kernel_elements *= k

            # 估算非零激活的数量
            # 这是一个简化估算，实际应用中可能需要更复杂的方法
            # 或者直接从模型中获取稀疏索引的统计信息
            active_voxels = indices.shape[0]

            # 计算MACs：每个激活点的MACs = 输入通道数 × 输出通道数 × 卷积核大小
            # 类似于PyTorch中的实现
            self.total_ops = active_voxels * in_channels * out_channels * kernel_elements

            print(f"SparseConvolution({self.name}) - 激活点: {active_voxels}, MACs: {self.total_ops}")

        def profile(self, intensors: [], outtensors: []):
            print(f"\nshape0: {intensors[0].shape}")
            indices = intensors[0].get_numpy()  # 稀疏索引
            values = intensors[1].get_numpy()  # 稀疏值
            input_shape = intensors[2].get_numpy()  # 输入张量的完整形状

            # 尝试获取权重
            if len(intensors) > 3:
                weight = intensors[3].get_numpy()  # 卷积核权重
            else:
                # 创建默认权重（仅用于形状推理，不用于实际计算）
                in_channels = values.shape[1]
                output_channels = 256
                kernel_size = self.kernel_size if hasattr(self, 'kernel_size') else [3, 3, 3]
                weight_shape = [output_channels, in_channels // self.groups, *kernel_size]
                weight = numpy.random.randn(*weight_shape).astype(values.dtype)
                print(f"SparseConvolution({self.name})缺少权重输入，使用随机权重进行形状推理")

            # 计算稀疏卷积的MACs
            print(f"\nshape1: {indices.shape[0]}")
            self._compute_sparse_conv_macs(indices, values, input_shape, weight)
            return [self.total_ops, 0]

    print("==================Start registering ScatterDense==================")

    @onnx_tool.NODE_REGISTRY.register()
    class ScatterDenseNode(onnx_tool.Node):
        def __init__(self, nodeproto):
            super().__init__(nodeproto)
            # 添加默认属性
            self.add_default_value('mode', 'update')  # 可选: 'update', 'add', 'mul'
            self.add_default_value('default_value', 0)

        def shape_infer(self, intensors: [], outtensors: []):
            """形状推理：根据output_shape输入确定输出形状"""
            # 检查输入数量
            if len(intensors) < 1:
                print(f"ScatterDense缺少必要的输入: {self.name}")
                return

            # 尝试获取output_shape输入
            if len(intensors) > 2:
                output_shape_tensor = intensors[2]
                try:
                    output_shape = output_shape_tensor.get_numpy().tolist()
                except:
                    # 如果无法获取实际值，尝试从属性或使用默认值
                    output_shape = [1, 128, 180, 180, 2]
                    print(f"ScatterDense({self.name})无法获取output_shape，使用默认值: {output_shape}")
            else:
                # 使用默认形状
                output_shape = [1, 128, 180, 180, 2]
                print(f"ScatterDense({self.name})缺少output_shape输入，使用默认形状: {output_shape}")

            # 更新输出张量的形状和数据类型
            outtensors[0].update_shape(output_shape)

            # 尝试设置数据类型
            if len(intensors) > 0:
                outtensors[0].update_dtype(intensors[0].dtype)
            else:
                outtensors[0].update_dtype(numpy.float32)  # 默认使用float32

        def value_infer(self, intensors: [], outtensors: []):
            """值推理：执行scatter操作，生成密集张量"""
            # 检查输入数量
            if len(intensors) < 1:
                print(f"ScatterDense缺少必要的输入: {self.name}")
                return

            try:
                # 获取indices
                if len(intensors) > 0:
                    indices = intensors[0].get_numpy()  # 稀疏索引
                else:
                    # 创建默认索引
                    indices = numpy.array([[0, 0, 0, 0]], dtype=numpy.int64)
                    print(f"ScatterDense({self.name})缺少indices输入，使用默认索引")

                # 获取values
                if len(intensors) > 1:
                    values = intensors[1].get_numpy()  # 稀疏值
                else:
                    # 创建默认值
                    values = numpy.array([[1.0]], dtype=numpy.float32)
                    print(f"ScatterDense({self.name})缺少values输入，使用默认值")

                # 获取output_shape
                if len(intensors) > 2:
                    output_shape = intensors[2].get_numpy().tolist()
                else:
                    output_shape = [1, 128, 180, 180, 2]
                    print(f"ScatterDense({self.name})缺少output_shape输入，使用默认形状")

                # 创建默认的密集张量
                dense_output = numpy.full(output_shape, self.default_value, dtype=values.dtype)

                # 根据mode执行scatter操作
                if self.mode == 'update':
                    # 直接更新值
                    dense_output[tuple(indices.T)] = values
                elif self.mode == 'add':
                    # 累加值
                    numpy.add.at(dense_output, tuple(indices.T), values)
                elif self.mode == 'mul':
                    # 累乘值
                    numpy.multiply.at(dense_output, tuple(indices.T), values)
                else:
                    print(f"不支持的scatter mode: {self.mode}")

                # 更新输出张量
                outtensors[0].update_tensor(dense_output)

            except Exception as e:
                print(f"无法执行ScatterDense的值推理: {e}")
                # 无法计算实际值时，至少设置形状
                self.shape_infer(intensors, outtensors)

        def profile(self, intensors: [], outtensors: []):
            """
                计算ScatterDense节点的MACs和参数数量
                返回: (macs, params)
            """
            # 获取输入和输出
            indices = intensors[0]

            # MACs计算：每个scatter操作通常涉及一次读取和一次写入
            # 因此可以近似为2倍的非零元素数量
            active_elements = indices.shape[1]
            macs = active_elements * 2  # 每个元素2次内存访问

            return [macs,0]

    print("==================Start getting onnx FLOPs==================")

    # 加载 ONNX 模型
    camera_backbone_path = args.path + "camera.backbone.onnx"
    camera_backbone_profile_path = args.export_path + "camera_backbone_profile.csv"
    camera_vtransform_path = args.path + "camera.vtransform.onnx"
    camera_vtransform_profile_path = args.export_path + "camera_vtransform_profile.csv"
    lidar_backbone_path = args.path + "lidar.backbone.xyz.onnx"
    lidar_backbone_profile_path = args.export_path + "lidar_backbone_profile.csv"
    fuser_path = args.path + "fuser.onnx"
    fuser_profile_path = args.export_path + "fuser_profile.csv"
    head_bbox_path = args.path + "head.bbox.onnx"
    head_bbox_profile_path = args.export_path + "head_bbox_profile.csv"

    img = torch.randn(1, 3, 256, 704)
    depth = torch.randn(1, 6, 256, 704)
    onnx_tool.model_profile(camera_backbone_path, {"img": img, "depth": depth},
                            save_profile=camera_backbone_profile_path)
    print("result of {} is saved to {}".format(camera_backbone_path, camera_backbone_profile_path))

    feat_in = torch.randn(1, 80, 360, 360)
    onnx_tool.model_profile(camera_vtransform_path, {"feat_in": feat_in},
                            save_profile=camera_vtransform_profile_path)
    print("result of {} is saved to {}".format(camera_vtransform_path, camera_vtransform_profile_path))

    # lidar_input = torch.randn(1, 5)
    # onnx_tool.model_profile(lidar_backbone_path, {"0": lidar_input},
    #                         save_profile=lidar_backbone_profile_path)
    # print("result of {} is saved to {}".format(lidar_backbone_path, lidar_backbone_profile_path))

    camera_feature = torch.randn(1, 80, 180, 180)
    lidar_feature = torch.randn(1, 256, 180, 180)
    onnx_tool.model_profile(fuser_path, {"camera_feature": camera_feature, "lidar_feature": lidar_feature},
                            save_profile=fuser_profile_path)
    print("result of {} is saved to {}".format(fuser_path, fuser_profile_path))

    middle = torch.randn(1, 512, 180, 180)
    onnx_tool.model_profile(head_bbox_path, {"middle": middle},
                            save_profile=head_bbox_profile_path)
    print("result of {} is saved to {}".format(head_bbox_path, head_bbox_profile_path))


if __name__ == "__main__":
    main()
