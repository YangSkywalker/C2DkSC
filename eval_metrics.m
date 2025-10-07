function [acc, nmi_val] = eval_metrics(true_labels, pred_labels)
% EVAL_METRICS 计算聚类结果的准确率(ACC)和归一化互信息(NMI)
%   输入:
%       true_labels - 真实标签向量
%       pred_labels - 预测标签向量
%   输出:
%       acc - 准确率
%       nmi_val - 归一化互信息

    % 检查输入有效性
    if length(true_labels) ~= length(pred_labels)
        error('真实标签和预测标签长度必须一致');
    end
    
    % 计算准确率(ACC)，调用getAC函数
    acc = getAC(pred_labels,true_labels)*100;
    
    % 计算归一化互信息(NMI)，调用nmi函数
    nmi_val = nmi(true_labels', pred_labels')*100;
end