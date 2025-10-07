% 示例：使用C2DkSCW106进行聚类并评估结果
% 假设已加载数据X和真实标签Y
load('Coil100_16_16matrixData.mat')
% 设置降维维度
dim = 12;

% 运行C2DkSCW106聚类算法
[pred_labels, WallCls] = C2DkSCW106(X, Y, dim);

% 计算评估指标
[acc, nmi_val] = eval_metrics(Y, pred_labels);

% 显示结果
fprintf('聚类准确率(ACC): %.4f\n', acc);
fprintf('归一化互信息(NMI): %.4f\n', nmi_val);