clear variables

% load training and test data
load('mat_data/train.mat')
load('mat_data/test.mat')

% compute Covariance matrix of features in training data
C = cov(Ytrain');

% compute the singular value decomposition (SVD) of the Covariance matrix 
[U, S, V] = svd(C);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Experiemnt for further exploring the impact of number of kept PCs for classification performance 

% set two zero vectors to store successful recognition
Dmax = 644; % explore dim = 1...644

F1 = zeros(Dmax,1);
F2 = zeros(Dmax,1);
time1 = zeros(Dmax,1);
time2 = zeros(Dmax,1);

% initial two zeros vectors (for pca and simple projection) to store the index of the training image 
% which is closest to each test image.
pred_pca = zeros(200,1);
pred_sp = zeros(200,1);

% experiments for PCA
for d = 1:Dmax
    tic;
    % compute PCA projection
    U1 = U(:,1:d);
    Y1 = U1' * Ytrain;
    i=1;
    for i = 1:200
        % randomly choose one image from test data
        I = Ytest(:,i); 
        I1=U1'*I;
        dist = zeros(200,1);
        % compute the distance between I1/I2 and every image in training data
        for j = 1:200
            dist(j) = norm(I1 - Y1(:,j), 2); 
        end
        % find the closest image with chosen image
        [minvalue1, index1] = min(dist);
        pred_pca(i) = index1;
        if ceil(index1/5) == ceil(i/5)
            F1(d) = F1(d) + 1;
        end      
    end
    time1(d) = toc;
end

% experiments for simple projection
for d = 1:Dmax
    tic;
    E = eye(644);
    % compute simple projection
    U2 = E(:,1:d); 
    Y2 = U2' * Ytrain; 
    i=1;
    for i = 1:200
        % randomly choose one image form test data
        I = Ytest(:,i); 
        I2=U2'*I;
        dist = zeros(200,1);
        % compute the distance between I1/I2 and every image in training data
        for j = 1:200
            dist(j) = norm(I2 - Y2(:,j), 2);
        end
        % find the closest image with chosen image
        [minvalue2, index2] = min(dist);
        pred_sp(i) = index2;
        if ceil(index2/5) == ceil(i/5)
            F2(d) = F2(d) + 1;
        end
    end
    time2(d) = toc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F1 = F1./200;
F2 = F2./200;

% Figure 1: test accuracy v.s. dim value
figure(1)
plot(F1);
hold on;
plot(F2, 'r');
xlabel('dim value')
ylabel('Recognition Accuracy of Test Data')
legend('PCA','Simple Projection','Location','southeast')

% Figure 2: computational time v.s. dim value
figure(2)
plot(time1);
hold on;
plot(time2, 'r');
xlabel('dim value')
ylabel('Time for labelling all test images')
legend('PCA','Simple Projection','Location','southeast')


% Figure 3: cumulative variance of accuracy v.s. dim value 
F1var = zeros(Dmax,1);
F2var = zeros(Dmax,1);
for i = 1:Dmax
    F1var(i,1) = var(F1(1:i,1));
    F2var(i,1) = var(F2(1:i,1));
end

figure(3)
plot(F1var);
hold on;
plot(F2var,'r')
xlabel('dim value')
ylabel('Variance of Recodniton Accuracy')
legend('PCA','Simple Projection')
