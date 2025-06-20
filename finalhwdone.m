clear all;
clc;
close all;

% 產生資料：400筆，x,y在[-0.8, 0.7]
x = (-0.8) + (0.7 + 0.8) * rand(1, 400);
y = (-0.8) + (0.7 + 0.8) * rand(1, 400);
z = 5 * sin(pi * x.^2) .* sin(2 * pi * y) + 1;  % 目標函數

% 資料切分
trainX1 = x(1:300);
trainX2 = y(1:300);
trainY = z(1:300);
testX1 = x(301:400);
testX2 = y(301:400);
testY = z(301:400);

% 正規化到 [0.2, 0.8]
a = 0.2; b = 0.8;
minZ = -4; maxZ = 6;
trainD = ((trainY - minZ) / (maxZ - minZ)) * (b - a) + a;
testD = ((testY - minZ) / (maxZ - minZ)) * (b - a) + a;

% 參數設定
n = 30;                   % 隱藏層神經元個數
eta = 0.5;                % 學習率
alpha = 0.9;              % 動量係數
N = 50000;                % 訓練次數

% 權重初始化
bias = rand(1,n)*20 - 10;
W1 = rand(1,n)*20 - 10;
W2 = rand(1,n)*20 - 10;
outputW = rand(1,n)*20 - 10;
outputbias = rand*20 - 10;

% 動量初始化
dbias_prev = zeros(1,n);
dW1_prev = zeros(1,n);
dW2_prev = zeros(1,n);
doutputW_prev = zeros(1,n);
doutputbias_prev = 0;

% 訓練開始
for t = 1:N
    % 前向傳遞
    hiddenV = W1'*trainX1 + W2'*trainX2 + bias';
    hiddenY = 1./(1+exp(-hiddenV));
    outputV = outputW*hiddenY + outputbias;
    outputY = 1./(1+exp(-outputV));
    
    % 誤差
    e = trainD - outputY;
    E = 0.5 * e.^2;
    
    % 反向傳播
    outputGrad = e .* outputY .* (1 - outputY);
    hiddenGrad = hiddenY .* (1 - hiddenY) .* (outputW' * outputGrad);
    
    % 計算 Batch 更新量
    for i = 1:300
        deltabias(:,i) = eta * hiddenGrad(:,i);
        deltaW1(:,i) = eta * hiddenGrad(:,i) * trainX1(i);
        deltaW2(:,i) = eta * hiddenGrad(:,i) * trainX2(i);
        deltaoutputW(:,i) = hiddenY(:,i) * outputGrad(:,i);  % 修正點
        deltaoutputbias(:,i) = eta * outputGrad(:,i);
    end
    
    % 動量 + 權重更新
    dbias = sum(deltabias,2)' / 300;
    dW1 = sum(deltaW1,2)' / 300;
    dW2 = sum(deltaW2,2)' / 300;
    doutputW = sum(deltaoutputW,2)' / 300;
    doutputbias = sum(deltaoutputbias,2)' / 300;
    
    bias = bias + dbias + alpha * dbias_prev;
    W1 = W1 + dW1 + alpha * dW1_prev;
    W2 = W2 + dW2 + alpha * dW2_prev;
    outputW = outputW + doutputW + alpha * doutputW_prev;
    outputbias = outputbias + doutputbias + alpha * doutputbias_prev;
    
    % 儲存目前動量
    dbias_prev = dbias;
    dW1_prev = dW1;
    dW2_prev = dW2;
    doutputW_prev = doutputW;
    doutputbias_prev = doutputbias;
    
    % 記錄平均誤差
    Eav(t) = mean(E);
end

% 測試資料預測
testHiddenV = W1'*testX1 + W2'*testX2 + bias';
testHiddenY = 1./(1+exp(-testHiddenV));
testOutputV = outputW * testHiddenY + outputbias;
testOutputY = 1./(1+exp(-testOutputV));

% 反量化回原始值
trainOutputY = (outputY - a) / (b - a) * (maxZ - minZ) + minZ;
testOutputY = (testOutputY - a) / (b - a) * (maxZ - minZ) + minZ;
trainRealY = (trainD - a) / (b - a) * (maxZ - minZ) + minZ;

% 畫圖 - 訓練誤差
figure(1)
plot(1:N, Eav);
xlabel('Iteration'); ylabel('E_{av}');
title('Training Error Convergence');

% 畫圖 - 合併兩個 surface 成一張圖
figure(2)
[XX1, YY1] = meshgrid(linspace(-0.8, 0.7, 50), linspace(-0.8, 0.7, 50));

% 預期輸出 Surface
ZZ1 = griddata(trainX1, trainX2, trainY, XX1, YY1, 'cubic');
subplot(1,2,1);
mesh(XX1, YY1, ZZ1);
title('Target Surface z = f(x,y)');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-0.8 0.7 -0.8 0.7 -4 6]);

% 訓練結果 Surface
ZZ2 = griddata(trainX1, trainX2, trainOutputY, XX1, YY1, 'cubic');
subplot(1,2,2);
mesh(XX1, YY1, ZZ2);
title('Trained Output Surface');
xlabel('x'); ylabel('y'); zlabel('z');
axis([-0.8 0.7 -0.8 0.7 -4 6]);

% 計算誤差並輸出
Etrain = mean(Eav);
etest = 0.5 * (testY - testOutputY).^2;
Etest = mean(etest);
fprintf('Average training error (Etrain): %.6f\n', Etrain);
fprintf('Average test error (Etest): %.6f\n', Etest);
