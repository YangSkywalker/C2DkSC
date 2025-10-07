function [pY,WallCls] = C2DkSCW106(X,Y,dim)


 
%%
% % % % Input:
%   Data.X: Data matrix. Each d1xd2 matrix of Data.X is a sample.
%   Data.Y: Data label vector.
%   FunPara.MaxIter: Maximum iteration number.
%   FunPara.epsilon: Parameter epsilon in capped l_{2,1}-norm.
%   FunPara.delta：Parameter need choose.
%   FunPara.Eta：Parameter need choose.
%   FunPara.Cta：Parameter need choose.

%   dim: reduced dimension.
%
% % % FunPara.Eta = 10^-8; % [10^-6:10^6]
% % % FunPara.Cta = 10^-8; % [10^-6:10^6]
% % % FunPara.epsilon = 10^0;  % F与G中<=的判断标准epsilon
% % % FunPara.MaxIter = 50;  
% % % FunPara.delta = 10^-5;   % F与G中防止分母出现0时所加的非负值

%%
Eta = 1; % 参数1；需要选取范围
Cta = 1; % 参数2；需要选取范围
MaxIter = 2;%20; % 设定最大迭代次数
quantile_val = 0.9; % 分位数设置为90%
delta = 10^-2; % 参数4；F与G中防止分母出现0时所加的非负值
[d1,d2,N] = size(X); % 获取矩阵的行数 列数 训练样本个数
label = unique(Y); % 将标签类别提取出来
c = length(label); % 计算训练样本类别数
N_i = zeros(c,1); % 用于存放每一类样本数量的矩阵
X_mean = mean(X,3); % 计算训练样本均值
W = eye(d1,dim); % 初始化W；得到d1xdim的单位矩阵
I = eye(d1,d1); % 初始化I；得到d1xdim的单位矩阵
Maxite=5;%10;%10;%00;%100; % For outer clustering iteration      
outite=0;
pY=crossvalind('Kfold',N,c);
% 初始化 epsilon1 和 epsilon2
epsilon1 = 1;
epsilon2 = 1;

    while outite<Maxite
        outite=outite+1;
        tY=pY;
        WallCls = cell(length(c));%每个类的投影；
        for k=1:c
            disc = ismember(k,unique(tY)); % 判断新类别是否有某一类别已经消失
            if disc ~= 1
                continue
            end
        A = [];
        D = [];
        %% 开始迭代 求解S1 S2
            for t = 1:MaxIter % 迭代次数 
                S1 = zeros(d1,d1); 
                S2 = zeros(d1,d1);
                H_w = [];
                H_b = [];
                F = [];
                G = [];
                % 初始化用于计算分位数的数组
                B_all = []; % 用于存储所有类间距离范数
                C_all = []; % 用于存储所有类内距离范数
                
                for h = 1:d2 % 训练样本的第h列
                    H_h_b = [];
                    H_h_w = [];
                    G_h = [];
                    F_h = [];
                    for i = k%1:c % 训练样本的第c类
                        X_i_Index = find(tY==i);
                        N_i = length(X_i_Index); % 第i类训练样本的个数
                        X_i = X(:,:,X_i_Index); % 第i个类对应的X数据
                        X_i_mean(:,:,i) = mean(X_i,3); % 第i个类的均值
                        H_h_i = N_i*(X_i_mean(:,h,i)-X_mean(:,h)); % 计算训练样本类间距离
                        H_h_b = [H_h_b,H_h_i]; % 得到类间散度矩阵
                        
                        % 计算类间距离范数并收集
                        norm_b = norm(N_i*W'*(X_i_mean(:,h,i)-X_mean(:,h)),2);
                        B_all = [B_all, norm_b];
                        
                        if norm_b <= epsilon1 % 使用当前epsilon1判断
                            Ind_1(i) = 1;
                        else
                            Ind_1(i) = 0;
                        end

                        G_h_i = Ind_1(i)/(norm_b + delta); % 计算权重值
                        G_h = [G_h,G_h_i]; % 得到权重矩阵G
                        
                        for s = 1:N_i % 第i类的第s个训练样本
                            % 计算类内距离范数并收集
                            norm_w = norm(W'*(X_i(:,h,s)-X_i_mean(:,h,i)),2);
                            C_all = [C_all, norm_w];
                            
                            if norm_w <= epsilon2 % 使用当前epsilon2判断
                                Ind_2(s) = 1;
                            else
                                Ind_2(s) = 0;
                            end

                            F_h_is = Ind_2(s)/(norm_w + delta); % 计算权重值
                            F_h = [F_h,F_h_is]; % 得到权重矩阵F
                            H_h_is = X_i(:,h,s)-X_i_mean(:,h,i); % 计算训练样本类内距离
                            H_h_w = [H_h_w,H_h_is]; % 得到类内散度矩阵
                        end
                    end
                    F_h = F_h'; % 得到所有训练样本第h列的权重矩阵F_h
                    G_h = G_h'; % 得到所有训练样本第h列的权重矩阵G_h
                    F = [F,F_h']; % 得到所有样本全部列的权重矩阵F
                    G = [G,G_h']; % 得到所有样本全部列的权重矩阵G
                    H_w = [H_w,H_h_w]; % 得到所有样本全部列的类内散度矩阵H_w
                    H_b = [H_b,H_h_b]; % 得到所有样本全部列的类间散度矩阵H_b
                end
                
                % 在每次迭代结束时更新 epsilon1 和 epsilon2 为90%分位数
                if ~isempty(B_all)
                    B_sorted = sort(B_all);
                    epsilon1 = B_sorted(round(length(B_sorted) * quantile_val));
                end
                
                if ~isempty(C_all)
                    C_sorted = sort(C_all);
                    epsilon2 = C_sorted(round(length(C_sorted) * quantile_val));
                end
                
                F = diag(F); % 得到对角化矩阵F
                G = diag(G); % 得到对角化矩阵G
                
                %% 求解W
                S1 = H_w*F*H_w'+Eta*I; % 得到加了权重后的类内散度矩阵
                S2 = H_b*G*H_b'+Cta*I; % 得到加了权重后的类间散度矩阵

                [W_best,D] = eig(S1,S2); 
                W_best = orth(W_best); % 让W变成正交单位矩阵
                D = diag(D); 
                eigIdx1 = find(D < 1e-5); % 找出近似于0的特征值；删除
                eigIdx = [eigIdx1]; 
                D(eigIdx) = [];
                W_best(:,eigIdx) = [];
                [~, index] = sort(D); % 从小到大排序
                W_best = W_best(:,index); % 提取特征值对应的特征向量
                D = D(index);
                W_best = W_best(:,1:min(dim,length(index))); % 确定W的维度；维度由S2的rank决定
                
                if t > 10 
                    if norm(W - W_best, 'fro') / norm(W, 'fro') < 1e-6
                        W = W_best;
                        break;
                    end
                end
                W = W_best;
            end
            WallCls{k}=W;
        end  
    end
    % 缺失的部分 - 应该根据投影后的距离重新分配标签
        for l = 1:N
            distances = zeros(c,1);
            for i = 1:c
                distances(i) = norm(WallCls{i}'*(X(:,:,l) - X_i_mean(:,:,i)), 'fro')^2;
            end
            [~, new_label] = min(distances);
            pY(l) = new_label;
        end
end