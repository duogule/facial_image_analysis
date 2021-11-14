clear variables

% load training and test data
load('mat_data/train.mat')
load('mat_data/test.mat')

% compute Covariance matrix of features in training data
C = cov(Ytrain');

% compute the singular value decomposition (SVD) of the Covariance matrix 
[U, S, V] = svd(C);

% set two zero matrix to store index of the images
index_train = zeros(5,5);
index_test = zeros(5,5);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Experiemnt for number of kept principle components and corresponding classification performance

% reduce the dimension of training iamges to multiple lower-dimensions (1, 3, 5, 7, 9) by PCA 
% and then perform kNN for classification on the reduced test data
for n = 1:5
    % compute PCA projection
    U1 = U(:,1:2*n-1); 
    Y1 = U1' * Ytrain;
    for i=1:5
        % randomly choose one image form test data
        index_test(i,n) = ceil(200 * rand(1)); 
        I = Ytest(:,index_test(i,n)); 
        I1 = U1'*I;
        dist = zeros(200,1);
        % compute the distance between I1 and every image in training data
        for j = 1:200
            dist(j) = norm(I1 - Y1(:,j), 2);
        end
        % find the closest image with chosen image
        [minvalue, index] = min(dist);
        index_train(i,n) = index;
    end
end 

% show random 5 images in test data and the prediction of PCA-kNN with different number of PCs kept
for i = 1:5
    figure(i);
    for j = 1:5
        subplot(2,5,j);
        I = reshape(Ytest(:,index_test(j,i)), 28, 23); 
        imagesc(I);
        colormap(gray);
        axis image;
        set(gca,'xtick',[],'ytick',[])

        subplot(2,5,j+5);
        I = reshape(Ytrain(:,index_train(j,i)), 28, 23); 
        imagesc(I);
        colormap(gray);
        axis image;
        set(gca,'xtick',[],'ytick',[])
        if ceil(index_test(j,i)/5) == ceil(index_train(j,i)/5)
           title('Correct!');
        end
    end
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
